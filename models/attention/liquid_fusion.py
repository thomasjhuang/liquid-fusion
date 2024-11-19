import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base_models import AttentionWithMetrics

class LiquidFusion(BaseAttention):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        attention_sink_size: int = 4,
        heavy_hitter_ratio: float = 0.2,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(num_heads, head_dim, dtype=dtype, **kwargs)
        self.attention_sink_size = attention_sink_size
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.dtype = dtype
        
        self.sink_cache = {'k': None, 'v': None}
        self.heavy_hitter_cache = {'k': None, 'v': None}
        self.accumulated_attention = None
        
        self.pattern_weights = nn.Parameter(
            torch.ones(num_heads, 3, dtype=self.dtype) / 3
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        hidden_states = hidden_states.to(self.dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.dtype)
            
        bsz, q_len, _ = hidden_states.size()
        self.heavy_hitter_size = max(1, int(q_len * self.heavy_hitter_ratio))

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Initialize sink cache
        if self.sink_cache['k'] is None:
            self.sink_cache = {
                'k': key_states[:, :, :self.attention_sink_size, :].detach().clone(),
                'v': value_states[:, :, :self.attention_sink_size, :].detach().clone()
            }

        scale = 1 / math.sqrt(self.head_dim)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        
        current_attention = scores.detach().sum(dim=1)
        if self.accumulated_attention is None:
            self.accumulated_attention = current_attention 
        else:
            self.accumulated_attention = current_attention

        # Select heavy hitter tokens
        _, top_indices = torch.topk(
            self.accumulated_attention.mean(dim=1),
            k=min(self.heavy_hitter_size, q_len),
            dim=-1
        )

        # Update heavy hitter cache
        gather_indices = top_indices.unsqueeze(1).unsqueeze(-1).expand(
            bsz, self.num_heads, -1, self.head_dim
        )
        
        self.heavy_hitter_cache = {
            'k': torch.gather(key_states, 2, gather_indices),
            'v': torch.gather(value_states, 2, gather_indices)
        }

        if attention_mask is not None:
            scores = scores + attention_mask

        # Calculate attention with consistent dtype
        attn_weights = F.softmax(scores.to(self.dtype), dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        # Add sink attention
        if self.sink_cache['k'] is not None:
            sink_scores = torch.matmul(query_states, self.sink_cache['k'].transpose(-2, -1)) * scale
            sink_weights = F.softmax(sink_scores.to(self.dtype), dim=-1)
            sink_output = torch.matmul(sink_weights, self.sink_cache['v'])
            attn_output = attn_output + sink_output * self.pattern_weights[:, 0].view(-1, 1, 1)

        # Add heavy hitter attention
        if self.heavy_hitter_cache['k'] is not None:
            heavy_scores = torch.matmul(query_states, self.heavy_hitter_cache['k'].transpose(-2, -1)) * scale
            heavy_weights = F.softmax(heavy_scores.to(self.dtype), dim=-1)
            heavy_output = torch.matmul(heavy_weights, self.heavy_hitter_cache['v'])
            attn_output = attn_output + heavy_output * self.pattern_weights[:, 1].view(-1, 1, 1)

        # Final projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if kwargs.get('output_attentions') else None, None