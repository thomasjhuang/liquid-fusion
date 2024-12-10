import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import math

class CombinedCache(nn.Module):
    def __init__(
        self,
        start_size: int,
        recent_size: int,
        heavy_ratio: float,
        recent_ratio: float,
    ):
        super().__init__()
        self.start_size = start_size  # StreamingLLM attention sinks
        self.recent_size = recent_size  # StreamingLLM recent tokens
        
        # H2O ratios for heavy hitter detection
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        
        # Initialize cache tracking
        self.attention_scores: Dict[int, torch.Tensor] = {}
        self.heavy_indices = None
        self.cache_indices = None
        
    def update(
        self,
        pre_rope_keys: torch.Tensor,
        values: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache using both StreamingLLM and H2O strategies"""
        bsz, num_heads, seq_len, head_dim = values.shape
        
        # Calculate cache sizes
        heavy_size = int(seq_len * self.heavy_ratio)
        recent_size = min(int(seq_len * self.recent_ratio), self.recent_size)
        
        # Keep attention sinks (StreamingLLM)
        sink_indices = torch.arange(self.start_size, device=values.device)
        
        # Identify heavy hitters (H2O)
        avg_attention = attn_weights.mean(dim=(0, 1))  # Average across batch and heads
        _, heavy_indices = torch.topk(avg_attention[self.start_size:], k=heavy_size)
        heavy_indices = heavy_indices + self.start_size  # Offset by sink tokens
        
        # Get recent indices
        recent_start = seq_len - recent_size
        recent_indices = torch.arange(recent_start, seq_len, device=values.device)
        
        # Combine indices
        keep_indices = torch.cat([
            sink_indices,
            heavy_indices,
            recent_indices
        ]).unique()
        
        # Update cache
        new_keys = pre_rope_keys[:, :, keep_indices, :]  # Use pre-RoPE keys (StreamingLLM)
        new_values = values[:, :, keep_indices, :]
        
        # Store attention scores for next update (H2O)
        self.attention_scores = attn_weights[:, :, :, keep_indices].detach()
        self.cache_indices = keep_indices
        
        return (new_keys, new_values)
    
    def get_position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get position IDs for the combined cache"""
        if self.cache_indices is None:
            return torch.arange(seq_len, device=device)
            
        # StreamingLLM: sinks keep original positions, rest use relative positions
        positions = torch.full((seq_len,), -1, device=device)
        positions[:self.start_size] = torch.arange(self.start_size, device=device)
        positions[self.start_size:] = torch.arange(
            seq_len - self.start_size, 
            device=device
        )
        return positions