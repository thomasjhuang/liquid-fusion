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
    """Apply rotary position embeddings to input x."""
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    
    # Make sure position_ids is 2D
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    
    # Get rotary embeddings for the requested positions
    cos = cos[position_ids % cos.size(0)]  # Add modulo to handle positions beyond seq_len
    sin = sin[position_ids % sin.size(0)]  # Add modulo to handle positions beyond seq_len
    
    # Add head dimension
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin.unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    # Make sure dimensions match
    if cos.size(2) > x.size(-2):
        cos = cos[:, :, :x.size(-2), :]
        sin = sin[:, :, :x.size(-2), :]
    elif cos.size(2) < x.size(-2):
        # Pad cos and sin to match x's sequence length
        pad_len = x.size(-2) - cos.size(2)
        cos = F.pad(cos, (0, 0, 0, pad_len, 0, 0, 0, 0))
        sin = F.pad(sin, (0, 0, 0, pad_len, 0, 0, 0, 0))
    
    return (x * cos) + (rotate_half(x) * sin)

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
        
        # Fix position handling for rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        # Ensure position_ids are properly aligned with the window size
        if past_key_value is not None and isinstance(past_key_value, DynamicCache):
            past_length = past_key_value.get_seq_length(self.layer_idx)
            # Ensure positions are continuous and properly offset
            position_ids = position_ids + past_length
            
            # Handle key positions separately
            key_position_ids = torch.arange(
                past_length, 
                past_length + key_states.size(-2), 
                device=position_ids.device
            )
            
            if key_position_ids.size(0) > self.window_size:
                # Create proper sink token positions that maintain relative positions
                sink_positions = torch.arange(
                    0, self.sink_size, 
                    device=position_ids.device
                )
                recent_positions = torch.arange(
                    past_length + key_states.size(-2) - (self.window_size - self.sink_size),
                    past_length + key_states.size(-2),
                    device=position_ids.device
                )
                key_position_ids = torch.cat([sink_positions, recent_positions])
        else:
            key_position_ids = position_ids.view(-1)

        key_position_ids = key_position_ids.unsqueeze(0)
        
        # Apply rotary embeddings with corrected positions
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

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
            # Ensure proper mask shape and values for streaming
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
            # Handle streaming window attention mask
            if attention_mask.size(-1) != key_states.size(-2):
                window_mask = torch.ones(
                    (bsz, 1, q_len, key_states.size(-2)),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                
                # Always allow attention to sink tokens
                window_mask[:, :, :, :self.sink_size] = 0
                
                # Add causal mask for recent tokens
                recent_size = min(self.window_size - self.sink_size, key_states.size(-2) - self.sink_size)
                window_mask[:, :, :, -recent_size:] = torch.triu(
                    torch.ones((q_len, recent_size), device=attention_mask.device),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(1)
                
                attention_mask = window_mask * -10000.0
            
            attn_weights = attn_weights + attention_mask

        # Calculate attention output
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Add safety checks for cache
        if past_key_value is not None and isinstance(past_key_value, DynamicCache):
            # Validate cache indices
            if self.layer_idx >= len(past_key_value.key_cache):
                past_key_value.key_cache.extend([None] * (self.layer_idx + 1 - len(past_key_value.key_cache)))
                past_key_value.value_cache.extend([None] * (self.layer_idx + 1 - len(past_key_value.value_cache)))
            
            # Ensure cache tensors are on correct device
            if past_key_value.key_cache[self.layer_idx] is not None:
                past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].to(hidden_states.device)
                past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].to(hidden_states.device)

        # Free memory explicitly
        if past_key_value is not None and isinstance(past_key_value, DynamicCache):
            past_key_value.key_cache[self.layer_idx] = past_key_value.key_cache[self.layer_idx].detach()
            past_key_value.value_cache[self.layer_idx] = past_key_value.value_cache[self.layer_idx].detach()

        return attn_output, attn_weights if output_attentions else None, past_key_value

def convert_to_streaming_attention(model, config):
    """Convert a model to use streaming attention."""
    device = next(model.parameters()).device
    
    # Ensure config has required attributes
    if not hasattr(config, 'window_size'):
        config.window_size = 256
    if not hasattr(config, 'sink_size'):
        config.sink_size = min(32, config.window_size // 8)
    
    # Validate config
    if config.sink_size >= config.window_size:
        raise ValueError(f"sink_size ({config.sink_size}) must be less than window_size ({config.window_size})")
    
    # Initialize cache with proper device placement
    model.past_key_values = DynamicCache()
    
    # Convert attention layers with proper weight transfer
    for i, layer in enumerate(model.model.layers):
        streaming_attn = StreamingAttention(config, layer_idx=i)
        
        # Ensure proper weight transfer
        with torch.no_grad():
            # Transfer weights safely
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                # Copy weights
                getattr(streaming_attn, name).weight.data.copy_(
                    getattr(layer.self_attn, name).weight.data
                )
                
                # Only copy bias if it exists in both models
                source_bias = getattr(getattr(layer.self_attn, name), 'bias', None)
                target_bias = getattr(getattr(streaming_attn, name), 'bias', None)
                if source_bias is not None and target_bias is not None:
                    target_bias.data.copy_(source_bias.data)
        
        # Transfer rotary embeddings
        streaming_attn.rotary_emb = layer.self_attn.rotary_emb
        
        # Ensure proper device placement
        streaming_attn = streaming_attn.to(device)
        layer.self_attn = streaming_attn
    
    return model