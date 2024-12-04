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
        if past_key_value is not None and isinstance(past_key_value, DynamicCache):
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        # Apply position-shifted rotary embeddings
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        
        if past_key_value is not None and isinstance(past_key_value, DynamicCache):
            if self.layer_idx < len(past_key_value.key_cache):
                past_key = past_key_value.key_cache[self.layer_idx]
                past_value = past_key_value.value_cache[self.layer_idx]
                
                if past_key.size(2) > 0:
                    # Calculate total length after concatenation
                    total_length = past_key.size(2) + key_states.size(2)
                    
                    # If total length exceeds window size, trim the cache
                    if total_length > self.window_size:
                        # Keep sink tokens (first sink_size tokens) and recent tokens
                        keep_length = self.window_size - key_states.size(2)
                        if keep_length > self.sink_size:
                            # Keep sink tokens and most recent tokens
                            past_key = torch.cat([
                                past_key[:, :, :self.sink_size],
                                past_key[:, :, -(keep_length-self.sink_size):]
                            ], dim=2)
                            past_value = torch.cat([
                                past_value[:, :, :self.sink_size],
                                past_value[:, :, -(keep_length-self.sink_size):]
                            ], dim=2)
                        else:
                            # Only keep sink tokens
                            past_key = past_key[:, :, :self.sink_size]
                            past_value = past_value[:, :, :self.sink_size]
                
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)

        # Apply position-shifted rotary embeddings to key states
        key_position_ids = torch.arange(key_states.size(-2), device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

        # Update cache if needed
        if use_cache:
            if isinstance(past_key_value, DynamicCache):
                # Ensure we don't exceed window size while preserving sink tokens
                if key_states.size(2) > self.window_size:
                    key_states = torch.cat([
                        key_states[:, :, :self.sink_size],
                        key_states[:, :, -(self.window_size-self.sink_size):]
                    ], dim=2)
                    value_states = torch.cat([
                        value_states[:, :, :self.sink_size],
                        value_states[:, :, -(self.window_size-self.sink_size):]
                    ], dim=2)
                past_key_value.update(key_states, value_states, self.layer_idx)
            else:
                past_key_value = (key_states, value_states)

        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Handle attention mask for streaming
            if attention_mask.size(-1) != key_states.size(-2):
                # If input attention mask is larger than window size, truncate it
                if attention_mask.size(-1) > self.window_size:
                    attention_mask = attention_mask[:, :, :, -self.window_size:]
                
                # Create new attention mask matching the key_states size
                new_attention_mask = torch.zeros(
                    bsz, 1, q_len, key_states.size(-2),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                
                # Place the attention mask in the correct position
                offset = key_states.size(-2) - attention_mask.size(-1)
                new_attention_mask[:, :, :, offset:] = attention_mask
                attention_mask = new_attention_mask
            
            attn_weights = attn_weights + attention_mask

        # Calculate attention output
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

def convert_to_streaming_attention(model, config):
    """Convert a model to use streaming attention."""
    device = next(model.parameters()).device  # Get model's device
    
    # Initialize cache for all layers using DynamicCache
    if not hasattr(model, 'past_key_values'):
        model.past_key_values = DynamicCache()
    
    # Convert attention layers
    for i, layer in enumerate(model.model.layers):
        streaming_attn = StreamingAttention(config, layer_idx=i)
        
        # Copy weights from original attention
        streaming_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
        streaming_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
        streaming_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()
        streaming_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data.clone()
        
        # Ensure rotary embeddings are on correct device
        streaming_attn.rotary_emb = layer.self_attn.rotary_emb
        streaming_attn = streaming_attn.to(device)
        
        # Replace the attention layer
        layer.self_attn = streaming_attn
        
    return model