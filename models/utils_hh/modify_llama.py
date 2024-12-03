import os
import copy
import math
import numpy as np 
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb
from transformers.cache_utils import Cache, DynamicCache, StaticCache

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']

class LlamaAttention_heavy_hitter(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.layer_idx = layer_idx
        
        # Heavy hitter specific attributes
        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio
        self.attention_masks_next = None 
        self.heavy_budget = None
        self.recent_budget = None
        self.cache_budget = None
        self.previous_scores = None
        self.input_length = []
        self.cache_budget_records = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        
        # Handle DynamicCache
        if past_key_value is not None:
            past_length = 0
            if hasattr(past_key_value, 'get_seq_length'):  # DynamicCache case
                past_length = past_key_value.get_seq_length()
                if past_length > 0:
                    key_states = torch.cat([past_key_value.key_cache, key_states], dim=2)
                    value_states = torch.cat([past_key_value.value_cache, value_states], dim=2)
            else:  # Regular tuple case
                past_length = past_key_value[0].shape[-2]
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            kv_seq_len += past_length

        # Get rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Update cache if needed
        if use_cache:
            if hasattr(past_key_value, 'update'):  # DynamicCache case
                past_key_value.update(key_states, value_states, self.layer_idx)
            else:  # Regular tuple case
                past_key_value = (key_states, value_states)

        # Calculate attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # Apply heavy hitter attention mask
        if self.attention_masks_next is not None:
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Calculate heavy hitter scores
        current_scores_sum = attn_weights.sum(0).sum(1)  # (heads, k-tokens)

        # Initialize budgets if not already set
        if self.cache_budget is None:
            self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
            self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
            self.cache_budget = self.heavy_budget + self.recent_budget
            self.cache_budget_records.append(self.cache_budget)
            self.input_length.append(attn_weights.shape[-1])

        if self.previous_scores is not None:
            current_scores_sum[:, :-1] += self.previous_scores
        
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum
        attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

        attn_tokens_all = self.previous_scores.shape[-1]
    
        # Ensure cache_budget is initialized before comparison
        if self.cache_budget is not None and attn_tokens_all > self.cache_budget:
            if self.recent_budget > 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = self.previous_scores[:, :-self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = self.previous_scores

            if self.heavy_budget > 0:
                _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        score_mask = attn_mask[:,:-1]
        score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = self.previous_scores * score_mask

        # Calculate output
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def convert_kvcache_llama_heavy_recent(model, config):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # Get layer index from name if it exists
            layer_idx = None
            if 'layers.' in name:
                try:
                    layer_idx = int(name.split('layers.')[-1].split('.')[0])
                except:
                    pass
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, config)
        if isinstance(module, LlamaAttention):
            model._modules[name] = LlamaAttention_heavy_hitter(config, layer_idx)
    return model