from dataclasses import dataclass

@dataclass
class AttentionConfig:
    num_heads: int = 32
    head_dim: int = 128
    max_position_embeddings: int = 2048

@dataclass
class LiquidFusionConfig(AttentionConfig):
    attention_sink_size: int = 4
    heavy_hitter_ratio: float = 0.2

@dataclass
class H2OConfig(AttentionConfig):
    cache_ratio: float = 0.2

@dataclass
class StreamingConfig(AttentionConfig):
    sink_size: int = 4