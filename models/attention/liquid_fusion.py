import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base_models import AttentionWithMetrics

class LiquidFusion(BaseLlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        # Use parent's implementation for Flash Attention
        if self.config._attn_implementation == "flash_attention_2":
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs
            )

        bsz, q_len, _ = hidden_states.size()
        
        # Get query/key/value states using parent's projections
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if provided
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Initialize or update sink cache
        if self.sink_cache['k'] is None:
            self.sink_cache = {
                'k': key_states[:, :, :self.attention_sink_size, :].detach().clone(),
                'v': value_states[:, :, :self.attention_sink_size, :].detach().clone()
            }
        else:
            alpha = 0.1  # Update rate
            self.sink_cache = {
                'k': (1 - alpha) * self.sink_cache['k'] + alpha * key_states[:, :, :self.attention_sink_size, :].detach(),
                'v': (1 - alpha) * self.sink_cache['v'] + alpha * value_states[:, :, :self.attention_sink_size, :].detach()
            }

        # Compute attention scores with sink tokens
        scale = 1 / math.sqrt(self.head_dim)
        
        # Regular attention
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
        
        # Add sink attention
        sink_scores = torch.matmul(query_states, self.sink_cache['k'].transpose(-2, -1)) * scale
        scores = torch.cat([sink_scores, scores], dim=-1)

        if attention_mask is not None:
            # Extend attention mask for sink tokens
            sink_mask = torch.zeros_like(sink_scores)
            attention_mask = torch.cat([sink_mask, attention_mask], dim=-1)
            scores = scores + attention_mask

        # Calculate attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Split attention weights for sink and regular tokens
        sink_weights, regular_weights = attn_weights.split([self.attention_sink_size, q_len], dim=-1)
        
        # Compute attention output
        sink_output = torch.matmul(sink_weights, self.sink_cache['v'])
        regular_output = torch.matmul(regular_weights, value_states)
        attn_output = sink_output + regular_output

        # Final projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value
