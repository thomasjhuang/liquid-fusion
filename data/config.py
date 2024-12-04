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
    attention_type: str = "default"  # Options: "default", "h2o", "sparse_fixed", "sparse_strided", "streaming"
    
    # General parameters
    max_tokens: int = 128
    temperature: float = 0.7
    
    # Dataset configuration
    datasets: List[DatasetConfig] = None
    
    # Sparse attention parameters
    window_size: Optional[int] = None
    stride: Optional[int] = None
    
    # H2O attention parameters
    heavy_ratio: Optional[float] = None
    recent_ratio: Optional[float] = None
    
    # Streaming attention parameters
    sink_size: Optional[int] = None
    sink_update_rate: Optional[float] = None

    def switch_dataset(self, dataset_name: str):
        """Switch datasets based on a string identifier."""
        dataset_configs = get_default_dataset_configs()
        if dataset_name in dataset_configs:
            self.datasets = [dataset_configs[dataset_name]]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(dataset_configs.keys())}")