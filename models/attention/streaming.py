import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache

__all__ = ['StreamingAttention', 'convert_to_streaming_attention']

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
        
        # Add dropout layer
        self.attention_dropout = config.attention_dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Metrics tracking
        self.input_length = []
        self.cache_budget_records = []
        
    def reset_cache(self):
        """Reset sink token cache"""
        self.sink_cache = {'k': None, 'v': None}
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Project queries, keys, values
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Track sequence lengths
        current_len = key_states.shape[-2]
        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        total_len = current_len + past_len + self.sink_size
        
        # Handle cache position
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0)
        
        # Get rotary embeddings with position_ids
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Initialize or update sink cache
        if self.sink_cache['k'] is None:
            self.sink_cache = {
                'k': key_states[:, :, :self.sink_size, :].detach().clone(),
                'v': value_states[:, :, :self.sink_size, :].detach().clone()
            }
        else:
            # Update sink tokens with exponential moving average
            self.sink_cache = {
                'k': (1 - self.sink_update_rate) * self.sink_cache['k'] + 
                     self.sink_update_rate * key_states[:, :, :self.sink_size, :].detach(),
                'v': (1 - self.sink_update_rate) * self.sink_cache['v'] + 
                     self.sink_update_rate * value_states[:, :, :self.sink_size, :].detach()
            }

        # Handle past key/value states
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        # Concatenate sink tokens with current tokens
        key_states = torch.cat([self.sink_cache['k'], key_states], dim=2)
        value_states = torch.cat([self.sink_cache['v'], value_states], dim=2)

        # Compute attention scores
        scale = 1 / math.sqrt(self.head_dim)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Create sliding window mask
        window_mask = torch.ones(
            bsz, self.num_heads, q_len, total_len, 
            dtype=torch.bool, 
            device=hidden_states.device
        )
        
        # Apply sliding window masking
        for i in range(q_len):
            start = max(self.sink_size, i + past_len - self.window_size // 2)
            end = min(total_len, i + past_len + self.window_size // 2 + 1)
            window_mask[:, :, i, start:end] = False
            window_mask[:, :, i, :self.sink_size] = False  # Always attend to sink tokens

        # Handle attention mask
        if attention_mask is not None:
            # Ensure attention mask has 4 dimensions [bsz, 1, q_len, k_len]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Create sink attention mask (all zeros = no masking)
            sink_mask = torch.zeros(
                bsz, 1, q_len, self.sink_size,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            
            # Pad attention mask to match total length
            pad_length = total_len - self.sink_size - attention_mask.size(-1)
            if pad_length > 0:
                attention_mask = F.pad(attention_mask, (0, pad_length), value=float("-inf"))
            
            # Concatenate sink mask with padded attention mask
            attention_mask = torch.cat([sink_mask, attention_mask], dim=-1)
            
            # Expand attention mask for all heads
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            
            window_mask = window_mask | attention_mask.bool()

        # Apply masks to attention scores
        attention_mask = window_mask.to(dtype=hidden_states.dtype) * -10000.0
        scores = scores + attention_mask

        # Calculate attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, (key_states, value_states) if use_cache else None

def convert_to_streaming_attention(model, config):
    """Convert a model to use streaming attention."""
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = StreamingAttention(config, layer_idx=i)
    return model