import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache

class LiquidFusion(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        if layer_idx is None:
            raise ValueError("layer_idx must be provided for LiquidFusion")
            
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        
        # Streaming config
        self.window_size = getattr(config, 'window_size', 256)
        self.sink_size = getattr(config, 'sink_size', 4)
        
        # Heavy-hitter config
        self.heavy_ratio = getattr(config, 'heavy_ratio', 0.2)
        self.recent_ratio = getattr(config, 'recent_ratio', 0.3)
        
        # Initialize sink cache
        self.sink_cache = {
            'k': None,  # Will be initialized on first forward pass
            'v': None
        }
        self.sink_update_rate = getattr(config, 'sink_update_rate', 0.1)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Get query/key/value states
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and isinstance(past_key_value, DynamicCache):
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)
            if self.layer_idx < len(past_key_value.key_cache):
                past_key = past_key_value.key_cache[self.layer_idx]
                past_value = past_key_value.value_cache[self.layer_idx]
                
                if past_key is not None and past_key.size(2) > 0:
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)

        # Update sink cache
        if self.sink_cache['k'] is not None and self.sink_cache['k'].size(0) != bsz:
            # Reset sink cache for new batch size
            self.sink_cache = {
                'k': None,
                'v': None
            }

        if self.sink_cache['k'] is None:
            self.sink_cache = {
                'k': key_states.new_zeros(bsz, self.num_heads, self.sink_size, self.head_dim),
                'v': value_states.new_zeros(bsz, self.num_heads, self.sink_size, self.head_dim)
            }
            if key_states.size(2) >= self.sink_size:
                self.sink_cache['k'].copy_(key_states[:, :, :self.sink_size, :].detach())
                self.sink_cache['v'].copy_(value_states[:, :, :self.sink_size, :].detach())
        else:
            # Ensure batch size matches
            if self.sink_cache['k'].size(0) != bsz:
                self.sink_cache = {
                    'k': self.sink_cache['k'].expand(bsz, -1, -1, -1),
                    'v': self.sink_cache['v'].expand(bsz, -1, -1, -1)
                }
            
            # Update with exponential moving average
            if key_states.size(2) >= self.sink_size:
                self.sink_cache = {
                    'k': (1 - self.sink_update_rate) * self.sink_cache['k'] + 
                         self.sink_update_rate * key_states[:, :, :self.sink_size, :].detach(),
                    'v': (1 - self.sink_update_rate) * self.sink_cache['v'] + 
                         self.sink_update_rate * value_states[:, :, :self.sink_size, :].detach()
                }

        # Compute attention scores
        scale = 1 / math.sqrt(self.head_dim)
        
        # Calculate attention with sink tokens
        sink_scores = torch.matmul(query_states, self.sink_cache['k'].transpose(-2, -1)) * scale
        regular_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Calculate heavy-hitter budget
        heavy_budget = max(1, int(self.heavy_ratio * key_states.size(-2)))
        recent_budget = max(1, int(self.recent_ratio * key_states.size(-2)))

        # Identify heavy hitters (global statistics approach)
        tmp_attn = F.softmax(regular_scores, dim=-1, dtype=torch.float32).to(regular_scores.dtype)
        tmp_sum = torch.sum(tmp_attn, dim=-2)
        _, heavy_indices = tmp_sum.topk(k=heavy_budget, dim=-1)

        # Create attention mask
        heavy_mask = torch.zeros_like(tmp_sum, dtype=torch.bool)
        heavy_mask.scatter_(-1, heavy_indices, True)
        heavy_mask = heavy_mask.unsqueeze(2).expand(-1, -1, regular_scores.size(-2), -1)

        # Add recent token mask
        recent_mask = torch.tril(torch.ones_like(regular_scores, dtype=torch.bool), diagonal=recent_budget)
        mask = torch.logical_or(heavy_mask, recent_mask)

        # Apply masks
        regular_scores = torch.where(mask, regular_scores, torch.finfo(regular_scores.dtype).min)
        
        # Ensure dimensions match before concatenation
        if sink_scores.size(-1) != self.sink_size:
            sink_scores = sink_scores[..., :self.sink_size]
        if regular_scores.size(-1) != key_states.size(-2):
            regular_scores = regular_scores[..., :key_states.size(-2)]
        
        scores = torch.cat([sink_scores, regular_scores], dim=-1)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure attention_mask has correct shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
            # Create sink attention mask
            sink_attention_mask = attention_mask.new_zeros(
                *attention_mask.shape[:-1], 
                self.sink_size
            )
            
            # Ensure regular attention mask matches key_states size
            if attention_mask.size(-1) != key_states.size(-2):
                attention_mask = attention_mask[..., :key_states.size(-2)]
            
            # Combine sink and regular attention masks
            attention_mask = torch.cat([sink_attention_mask, attention_mask], dim=-1)
            
            # Apply attention mask
            scores = scores + attention_mask

        # Calculate attention weights and output
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Split attention for sink and regular tokens
        sink_weights, regular_weights = attn_weights.split([self.sink_size, key_states.size(-2)], dim=-1)
        
        # Compute attention output
        sink_output = torch.matmul(sink_weights, self.sink_cache['v'])
        regular_output = torch.matmul(regular_weights, value_states)
        attn_output = sink_output + regular_output

        # Final projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Update cache if needed
        if use_cache:
            if isinstance(past_key_value, DynamicCache):
                past_key_value.update(key_states, value_states, self.layer_idx)
            else:
                past_key_value = (key_states, value_states)

        return attn_output, attn_weights if output_attentions else None, past_key_value

def convert_to_liquid_fusion(model, config):
    """Convert a model to use liquid fusion attention."""
    device = next(model.parameters()).device
    
    # Validate and set config defaults
    if not hasattr(config, 'window_size'):
        config.window_size = 256
    if not hasattr(config, 'sink_size'):
        config.sink_size = 4
    if not hasattr(config, 'heavy_ratio'):
        config.heavy_ratio = 0.2
    if not hasattr(config, 'recent_ratio'):
        config.recent_ratio = 0.3
    if not hasattr(config, 'sink_update_rate'):
        config.sink_update_rate = 0.1
    
    # Convert attention layers
    for i, layer in enumerate(model.model.layers):
        liquid_attn = LiquidFusion(config, layer_idx=i)
        
        # Transfer weights
        with torch.no_grad():
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                getattr(liquid_attn, name).weight.data.copy_(
                    getattr(layer.self_attn, name).weight.data
                )
        
        # Transfer rotary embeddings
        liquid_attn.rotary_emb = layer.self_attn.rotary_emb
        
        # Ensure proper device placement
        liquid_attn = liquid_attn.to(device)
        layer.self_attn = liquid_attn
    
    return model
