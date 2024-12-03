from transformers import LlamaConfig
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class AttentionConfig:
    num_heads: int = 32
    head_dim: int = 128
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0
    attention_bias: bool = False
    # Common parameters for all attention variants
    hidden_size: int = 4096  # num_heads * head_dim
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # For grouped query attention (GQA)
    rope_theta: float = 10000.0
    attention_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"

@dataclass 
class LlamaBaseAttentionConfig(AttentionConfig):
    """Configuration for standard LlamaAttention (eager implementation)"""
    attention_implementation: Literal["eager"] = "eager"
    # No additional parameters needed - uses basic matrix multiplication

@dataclass
class LlamaSdpaAttentionConfig(AttentionConfig):
    """Configuration for Scaled Dot Product Attention variant"""
    attention_implementation: Literal["sdpa"] = "sdpa"
    # SDPA specific parameters
    is_causal: bool = True
    scale: float = None  # If None, uses 1/sqrt(head_dim)

@dataclass
class LlamaFlashAttention2Config(AttentionConfig):
    """Configuration for Flash Attention 2 variant"""
    attention_implementation: Literal["flash_attention_2"] = "flash_attention_2"
    # Flash Attention 2 specific parameters
    softmax_scale: float = None  # If None, uses 1/sqrt(head_dim)
    attention_dropout: float = 0.0
    is_causal: bool = True

@dataclass
class LiquidFusionConfig(LlamaConfig):
    """Configuration for all attention variants"""
    # Common attention parameters
    attention_sink_size: int = 4
    heavy_hitter_ratio: float = 0.2
    window_size: int = 256
    cache_ratio: float = 0.2
    
    # Attention implementation selection
    attention_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    
    def __post_init__(self):
        super().__init__()
        # Ensure config has all required fields
        if not hasattr(self, 'num_attention_heads'):
            self.num_attention_heads = 32
        if not hasattr(self, 'hidden_size'):
            self.hidden_size = self.num_attention_heads * 128  # head_dim