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
    def __init__(self, config: LlamaConfig, layer_idx: int):
        if layer_idx is None:
            raise ValueError("layer_idx must be provided for LlamaAttention_heavy_hitter")
            
        super().__init__(config)
        self.layer_idx = layer_idx
        
        # Heavy hitter specific attributes
        self.heavy_budget_ratio = getattr(config, 'heavy_ratio', 0.5)  # Default to 0.5 if not specified
        self.recent_budget_ratio = getattr(config, 'recent_ratio', 0.5)  # Default to 0.5 if not specified
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
            if isinstance(past_key_value, DynamicCache):
                # Ensure the layer index exists in the cache
                while len(past_key_value.key_cache) <= self.layer_idx:
                    past_key_value.key_cache.append([])
                    past_key_value.value_cache.append([])
                
                # Only concatenate if there's cached data for this layer
                if past_key_value.key_cache[self.layer_idx] != []:
                    key_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=2)
                    value_states = torch.cat([past_key_value.value_cache[self.layer_idx], value_states], dim=2)
                    kv_seq_len = key_states.shape[-2]
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

        # Handle score accumulation with different sizes
        if self.previous_scores is not None:
            # Resize previous scores if needed
            if self.previous_scores.size(-1) != current_scores_sum.size(-1) - 1:
                # Create new tensor of correct size
                new_scores = torch.zeros_like(current_scores_sum[:, :-1])
                # Copy as much of the old scores as possible
                min_size = min(new_scores.size(-1), self.previous_scores.size(-1))
                new_scores[:, -min_size:] = self.previous_scores[:, -min_size:]
                self.previous_scores = new_scores
            
            current_scores_sum[:, :-1] += self.previous_scores
        
        dtype_attn_weights = attn_weights.dtype
        attn_weights_devices = attn_weights.device
        assert attn_weights.shape[0] == 1
        self.previous_scores = current_scores_sum[:, :-1].clone()  # Store all but the last token

        # Apply heavy hitter attention mask
        if self.attention_masks_next is not None:
            # Ensure mask matches the attention weights size
            if self.attention_masks_next.size(-1) != attn_weights.size(-1):
                # Create new mask of correct size
                attn_mask = torch.ones(current_scores_sum.shape[0], attn_weights.size(-1)).to(dtype_attn_weights).to(attn_weights_devices)
                if attn_weights.size(-1) > self.cache_budget:
                    if self.recent_budget > 0:
                        attn_mask[:, :-self.recent_budget] = 0
                        if self.heavy_budget > 0:
                            selected_set = current_scores_sum[:, :-self.recent_budget]
                            _, keep_topk = selected_set.topk(k=min(self.heavy_budget, selected_set.size(-1)), dim=-1, largest=True)
                            attn_mask = attn_mask.scatter(-1, keep_topk, 1)
                self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)
            
            attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

        attn_mask = torch.ones(
            current_scores_sum.shape[0], 
            current_scores_sum.shape[1]
        ).to(dtype_attn_weights).to(attn_weights_devices)
        
        attn_tokens_all = current_scores_sum.shape[-1]

        # Apply heavy hitter and recent token masking
        if self.cache_budget is not None and attn_tokens_all > self.cache_budget:
            if self.recent_budget > 0:
                attn_mask[:, :-self.recent_budget] = 0
                selected_set = current_scores_sum[:, :-self.recent_budget]
            else:
                attn_mask[:, :] = 0
                selected_set = current_scores_sum

            if self.heavy_budget > 0:
                _, keep_topk = selected_set.topk(k=min(self.heavy_budget, selected_set.size(-1)), dim=-1, largest=True)
                attn_mask = attn_mask.scatter(-1, keep_topk, 1)

        self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)

        # Update previous scores with mask
        score_mask = attn_mask.clone()  # Use same size as current scores
        if self.recent_budget > 0:
            score_mask[:, -self.recent_budget:] = 1
        self.previous_scores = current_scores_sum * score_mask

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
    """Convert model to use heavy hitter attention with proper layer indices."""
    def _convert_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LlamaAttention):
                # Extract layer index from the full name
                layer_idx = None
                try:
                    if 'layers.' in full_name:
                        layer_idx = int(full_name.split('layers.')[-1].split('.')[0])
                except ValueError as e:
                    print(f"Warning: Could not extract layer_idx from {full_name}")
                    continue  # Skip conversion if we can't determine layer index
                
                if layer_idx is not None:
                    # Create new attention layer with explicit layer_idx
                    new_attention = LlamaAttention_heavy_hitter(config, layer_idx=layer_idx)
                    # Copy existing weights
                    new_attention.load_state_dict(child.state_dict())
                    # Update module
                    setattr(module, name, new_attention)
            
            elif len(list(child.children())) > 0:
                _convert_module(child, full_name)
                
        return module

    return _convert_module(model)