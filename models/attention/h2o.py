import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_models import AttentionWithMetrics
import math

class H2OAttention(BaseAttention):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        cache_ratio: float = 0.2,
        **kwargs
    ):
        super().__init__(num_heads, head_dim, **kwargs)
        self.cache_ratio = cache_ratio
        self.token_cache = {'k': None, 'v': None}
        self.importance_scores = None
        
    def reset_cache(self):
        """Reset the token cache and importance scores"""
        self.token_cache = {'k': None, 'v': None}
        self.importance_scores = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        # Reset cache if sequence length changes
        if self.token_cache['k'] is not None:
            if self.token_cache['k'].size(2) != int(q_len * self.cache_ratio):
                self.reset_cache()

        # Project query, key, value
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate importance scores based on key norm
        current_importance = torch.norm(key_states, dim=-1)
        if self.importance_scores is None or self.importance_scores.size(-1) != q_len:
            self.importance_scores = current_importance
        else:
            self.importance_scores = 0.9 * self.importance_scores + 0.1 * current_importance

        # Select top-k tokens based on importance
        cache_size = max(1, int(q_len * self.cache_ratio))
        _, top_indices = torch.topk(self.importance_scores.mean(1), k=cache_size, dim=-1)

        # Update token cache
        gather_indices = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
        cached_keys = torch.gather(key_states, 2, gather_indices)
        cached_values = torch.gather(value_states, 2, gather_indices)

        # Calculate attention scores
        scale = 1 / math.sqrt(self.head_dim)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask

        # Calculate attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)

        # Update cache after computing main attention
        self.token_cache = {
            'k': cached_keys.detach(),
            'v': cached_values.detach()
        }

        # Add cached token attention if cache exists
        if self.token_cache['k'] is not None:
            cache_scores = torch.matmul(query_states, self.token_cache['k'].transpose(-2, -1)) * scale
            cache_weights = F.softmax(cache_scores, dim=-1)
            cache_output = torch.matmul(cache_weights, self.token_cache['v'])
            attn_output = attn_output + cache_output

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if kwargs.get('output_attentions') else None, None
