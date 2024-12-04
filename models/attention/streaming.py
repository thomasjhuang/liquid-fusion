import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class StreamingAttention(LlamaAttention):
    """Streaming attention with sink tokens, based on StreamingLLM paper"""
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        if layer_idx is None:
            raise ValueError("layer_idx must be provided for StreamingAttention")
            
        super().__init__(config)
        self.layer_idx = layer_idx
        
        # Streaming specific attributes
        self.window_size = getattr(config, 'window_size', 256)
        self.sink_size = getattr(config, 'sink_size', 32)
        self.sink_update_rate = getattr(config, 'sink_update_rate', 0.1)
        
        # Initialize sink cache
        self.sink_cache = {'k': None, 'v': None}
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, DynamicCache):
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # Handle position IDs
        if position_ids is None:
            position_ids = torch.arange(0, q_len, device=hidden_states.device).unsqueeze(0)

        # Apply rotary embeddings (using transformers' built-in functions)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key/value states
        if past_key_value is not None:
            if isinstance(past_key_value, DynamicCache):
                past_k, past_v = past_key_value.get_key_value(self.layer_idx)
                if past_k is not None:
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Update cache if needed
        if use_cache:
            if isinstance(past_key_value, DynamicCache):
                past_key_value.update(key_states, value_states, self.layer_idx)
            else:
                past_key_value = (key_states, value_states)

        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # Calculate attention output
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

def convert_to_streaming_attention(model, config):
    """Convert a model to use streaming attention."""
    num_layers = len(model.model.layers)
    
    # Initialize cache for all layers using DynamicCache
    if not hasattr(model, 'past_key_values'):
        model.past_key_values = DynamicCache()
        # DynamicCache will automatically handle layer initialization
    
    # Convert attention layers
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = StreamingAttention(config, layer_idx=i)
        
    return model