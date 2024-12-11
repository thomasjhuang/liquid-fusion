from types import MethodType
from typing import Optional, Tuple, List, Any, Dict, Union
import torch
from torch import nn, Tensor
import math
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv, rotate_half
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def liquid_fusion_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[DynamicCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    
    # Project queries, keys and values
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
    
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    is_decode = q_len == 1
    
    if is_decode:
        if use_cache:
            # For decoding, use the standard cache update
            key_states_to_cache = repeat_kv(key_states, self.num_key_value_groups)
            value_states_to_cache = repeat_kv(value_states, self.num_key_value_groups)
            
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states_to_cache, value_states_to_cache,
                self.layer_idx,
                cache_kwargs=cache_kwargs
            )
            
            kv_seq_len = key_states.shape[2]
            position_ids = torch.tensor(kv_seq_len-1, dtype=torch.long, device=position_ids.device).unsqueeze(0).unsqueeze(0)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if use_cache:
            # During prefill, cache both regular KV states and identify heavy hitters
            key_states_to_cache = repeat_kv(key_states, self.num_key_value_groups)
            value_states_to_cache = repeat_kv(value_states, self.num_key_value_groups)
            
            # Calculate attention scores for heavy hitter identification
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            temp_attn = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # Select tokens to keep: sink tokens (first few) and heavy hitters
            sink_size = self.sink_size
            
            # Calculate attention scores per head
            if q_len < 8192:
                attn_scores = temp_attn.detach().sum(0).sum(1)  # [heads, k-tokens]
            else:
                attn_scores = torch.zeros(
                    self.num_attention_heads, kv_seq_len, 
                    dtype=temp_attn.dtype, 
                    device=temp_attn.device
                )
                # Split across heads for memory efficiency
                attn_scores[:self.num_attention_heads // 2, :] = temp_attn[:, :self.num_attention_heads // 2, :, :].sum(0).sum(1)
                attn_scores[self.num_attention_heads // 2:, :] = temp_attn[:, self.num_attention_heads // 2:, :, :].sum(0).sum(1)
            
            # Keep sink tokens
            sink_keys = key_states_to_cache[:, :, :sink_size, :]
            sink_values = value_states_to_cache[:, :, :sink_size, :]
            
            # Calculate remaining tokens and heavy size
            remaining_tokens = max(0, attn_scores.shape[-1] - sink_size)
            heavy_size = min(int(self.heavy_budget), remaining_tokens)
            
            if heavy_size > 0:
                # Get top-k indices across all heads
                _, heavy_indices = torch.topk(attn_scores.sum(0)[sink_size:], k=heavy_size)
                heavy_indices += sink_size
                
                # Keep heavy hitter tokens
                heavy_keys = key_states_to_cache[:, :, heavy_indices, :]
                heavy_values = value_states_to_cache[:, :, heavy_indices, :]
                
                # Combine parts
                key_states_to_cache = torch.cat([sink_keys, heavy_keys], dim=2)
                value_states_to_cache = torch.cat([sink_values, heavy_values], dim=2)
            else:
                key_states_to_cache = sink_keys
                value_states_to_cache = sink_values
            
            cache_kwargs = {"sin": sin, "cos": cos}
            past_key_value.update(key_states_to_cache, value_states_to_cache, self.layer_idx, cache_kwargs=cache_kwargs)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if attention_mask is not None:
        attention_mask = attention_mask.expand(-1, -1, q_len, -1)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :attn_weights.shape[-2], :attn_weights.shape[-1]]

        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    
    attn_weights = attn_weights.softmax(dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value if use_cache else None


def convert_to_liquid_fusion(
    model, 
    sink_size: int = 4,
    heavy_budget: int = 50, 
    recent_budget: int = 50
):
    """Convert a model to use liquid fusion attention."""
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = convert_to_liquid_fusion(
                module, sink_size, heavy_budget, recent_budget
            )
            
        if isinstance(module, LlamaAttention):
            module.forward = MethodType(liquid_fusion_forward, module)
            module.sink_size = sink_size
            module.heavy_budget = heavy_budget
            module.recent_budget = recent_budget
            module.cache_budget = sink_size + heavy_budget + recent_budget
            
    return model