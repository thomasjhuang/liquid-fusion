from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

@dataclass
class DatasetConfig:
    name: str
    splits: List[str]
    config: Optional[str] = None
    input_prefix: str = ""  # e.g., "Context: ", "Question: "
    output_prefix: str = ""  # e.g., "Answer: "
    max_samples: Optional[int] = None

def get_default_dataset_configs() -> Dict[str, DatasetConfig]:
    """Get default dataset configurations with appropriate formatting"""
    return {
        "hellaswag": DatasetConfig(
            name="hellaswag",
            splits=["validation"],
            input_prefix="Context: ",
            output_prefix="Answer: "
        ),
        "piqa": DatasetConfig(
            name="piqa",
            splits=["validation"],
            input_prefix="Goal: ",
            output_prefix="Answer: "
        ),
        "mmlu": DatasetConfig(
            name="cais/mmlu",
            splits=["validation"],
            input_prefix="Question: ",
            output_prefix="Answer: "
        ),
    }

@dataclass
class BenchmarkConfig:
    def __init__(
        self,
        model_name: str,
        model_type: str = "llama",
        # Core settings
        device: str = "cuda",
        dtype: str = "bfloat16",
        # Generation settings
        max_tokens: int = 8,
        temperature: float = 0.1,
        # Sequence handling
        sequence_length: int = 256,
        max_position_embeddings: int = 2048,
        # StreamingLLM specific
        start_size: int = 4,
        recent_size: int = 64,
        # Sparse attention specific
        window_size: int = 256,
        stride: int = 128,
        # Dataset config
        datasets: List[DatasetConfig] = None,
        # Sampling control
        max_samples: Optional[int] = None,
        # Strategy control
        strategies: List[str] = None,
        # Metrics control
        enable_cache_metrics: bool = False
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.sequence_length = sequence_length
        self.max_position_embeddings = max_position_embeddings
        self.start_size = start_size
        self.recent_size = recent_size
        self.window_size = window_size
        self.stride = stride
        self.max_samples = max_samples
        self.strategies = strategies or ["full"]
        self.enable_cache_metrics = enable_cache_metrics
        self.datasets = datasets or [
            DatasetConfig(
                name="super_glue",
                config="copa",
                splits=["validation"],
                input_prefix="Given the premise: ",
                output_prefix="Answer: "
            )
        ]