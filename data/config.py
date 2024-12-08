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
        model_type: str,
        attention_type: str = "default",
        # General parameters
        max_tokens: int = 128,
        temperature: float = 0.7,
        dtype: str = "float16",
        device: str = "cuda",
        sequence_length: int = 512,
        
        # Dataset configuration
        datasets: List[DatasetConfig] = None,
        
        # Sparse attention parameters
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        
        # H2O attention parameters
        heavy_ratio: Optional[float] = None,
        recent_ratio: Optional[float] = None,
        
        # Streaming attention parameters
        sink_size: Optional[int] = None,
        sink_update_rate: Optional[float] = None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.attention_type = attention_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.device = device
        self.sequence_length = sequence_length
        self.datasets = datasets
        self.window_size = window_size
        self.stride = stride
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.sink_size = sink_size
        self.sink_update_rate = sink_update_rate

    def switch_dataset(self, dataset_name: str):
        """Switch datasets based on a string identifier."""
        dataset_configs = get_default_dataset_configs()
        if dataset_name in dataset_configs:
            self.datasets = [dataset_configs[dataset_name]]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(dataset_configs.keys())}")