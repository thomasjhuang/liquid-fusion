from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    name: str
    splits: List[str]
    config: Optional[str] = None
    revision: Optional[str] = None
    input_field: str = "document"
    reference_field: str = "summary"
    input_prefix: str = "Summarize the following document:\n\n"
    output_prefix: str = "\n\nSummary:"
    max_samples: Optional[int] = None


@dataclass
class BenchmarkConfig:
    model_name: str
    model_revision: Optional[str] = None
    model_type: str = "llama"
    device: str = "cuda"
    dtype: str = "float16"
    attention_type: str = "default"
    max_new_tokens: int = 128
    temperature: float = 0.0
    sequence_length: int = 4096
    max_position_embeddings: int = 4096
    strategies: List[str] = field(default_factory=lambda: ["full"])
    sink_size: int = 4
    start_size: int = 4
    recent_size: int = 1024
    heavy_budget: int = 512
    recent_budget: int = 512
    datasets: List[DatasetConfig] = field(
        default_factory=lambda: [
            DatasetConfig(
                name="EdinburghNLP/xsum",
                splits=["test"],
            )
        ]
    )
    max_samples: int = 10
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.max_new_tokens < 1 or self.sequence_length < 1:
            raise ValueError("Token limits must be positive")
        if self.max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.sink_size < 0 or self.heavy_budget < 0 or self.recent_budget < 1:
            raise ValueError("Invalid LiquidFusion cache budget")