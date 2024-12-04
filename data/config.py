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
    model_name: str
    model_type: str
    attention_type: str = "default"  # Options: "default", "h2o", "sparse_fixed", "sparse_strided"
    
    # H2O specific parameters
    heavy_ratio: float = 0.1  # Ratio of tokens to keep as heavy hitters
    recent_ratio: float = 0.1  # Ratio of recent tokens to keep
    
    # Sparse attention specific parameters
    window_size: int = 256  # Window size for local attention
    stride: int = 128  # Stride size for strided attention

    datasets: List[DatasetConfig] = field(default_factory=list)
    sequence_length: int = 1024
    dtype: str = "bfloat16"
    device: str = "cuda"
    temperature: float = 0.7
    max_tokens: int = 50 
    use_amp: bool = False

    def switch_dataset(self, dataset_name: str):
        """Switch datasets based on a string identifier."""
        dataset_configs = get_default_dataset_configs()
        if dataset_name in dataset_configs:
            self.datasets = [dataset_configs[dataset_name]]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(dataset_configs.keys())}")