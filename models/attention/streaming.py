import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_models import AttentionWithMetrics
import math

class StreamingAttention(BaseLlamaAttention):
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
        # Use Flash Attention when available
        if self.config._attn_implementation == "flash_attention_2":
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                sliding_window=self.window_size,  # Enable sliding window in Flash Attention
                **kwargs
            )
        if self.sink_cache['k'] is None:
            self.reset_cache()
            
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Initialize or update sink cache
        if self.sink_cache['k'] is None:
            self.sink_cache = {
                'k': key_states[:, :, :self.sink_size, :].detach().clone(),
                'v': value_states[:, :, :self.sink_size, :].detach().clone()
            }
        else:
            alpha = 0.1
            self.sink_cache = {
                'k': (1 - alpha) * self.sink_cache['k'] + alpha * key_states[:, :, :self.sink_size, :].detach(),
                'v': (1 - alpha) * self.sink_cache['v'] + alpha * value_states[:, :, :self.sink_size, :].detach()
            }

        # Concatenate sink tokens with current tokens
        key_states = torch.cat([self.sink_cache['k'], key_states], dim=2)
        value_states = torch.cat([self.sink_cache['v'], value_states], dim=2)

        # Compute attention scores
        scale = 1 / math.sqrt(self.head_dim)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        # Create sliding window mask
        total_len = q_len + self.sink_size
        window_mask = torch.ones(bsz, self.num_heads, q_len, total_len, dtype=torch.bool, device=hidden_states.device)
        
        for i in range(q_len):
            start = max(self.sink_size, i - self.window_size // 2)
            end = min(total_len, i + self.window_size // 2 + 1)
            window_mask[:, :, i, start:end] = False
            window_mask[:, :, i, :self.sink_size] = False

        # Apply attention mask
        if attention_mask is not None:
            # Reshape attention mask from [bsz, 1, 1, 1, q_len, q_len] to [bsz, q_len, q_len]
            attention_mask = attention_mask.squeeze(1).squeeze(1).squeeze(1)  # [bsz, q_len, q_len]
            
            # Create sink attention
            sink_attention = torch.ones(bsz, q_len, self.sink_size, device=attention_mask.device)
            
            # Concatenate sink attention with regular attention mask
            extended_attention_mask = torch.cat([sink_attention, attention_mask], dim=-1)  # [bsz, q_len, total_len]
            
            # Expand for all heads
            extended_attention_mask = extended_attention_mask.unsqueeze(1).expand(bsz, self.num_heads, q_len, total_len)
            
            # Combine with window mask
            window_mask = window_mask | extended_attention_mask.bool()

        # Convert mask to attention values
        attention_mask = window_mask.to(dtype=hidden_states.dtype) * -10000.0
        scores = scores + attention_mask

        # Calculate attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            attn_weights = attn_weights.permute(1, 2, 0, 3).unsqueeze(0).unsqueeze(0)
            
        return attn_output, attn_weights if output_attentions else None, None
