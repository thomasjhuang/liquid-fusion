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
        max_new_tokens: int = 128,  # For summarization
        temperature: float = 0.0,
        # Sequence length settings
        sequence_length: int = 4096,  # Increased for long documents
        max_position_embeddings: int = 4096,
        # Attention strategy settings
        strategies: List[str] = None,
        # StreamingLLM specific
        start_size: int = 4,
        recent_size: int = 3496,  # Following StreamingLLM paper
        # H2O specific
        heavy_budget: int = 800,    # ~20% for heavy hitters
        recent_budget: int = 800,   # ~20% for recent tokens
        # Dataset config
        datasets: List[DatasetConfig] = None,
        max_samples: Optional[int] = None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.sequence_length = sequence_length
        self.max_position_embeddings = max_position_embeddings
        self.strategies = strategies or ["base"]
        # StreamingLLM
        self.start_size = start_size
        self.recent_size = recent_size
        # H2O
        self.heavy_budget = heavy_budget
        self.recent_budget = recent_budget
        self.datasets = datasets or [
            DatasetConfig(
                name="ccdv/pubmed-summarization",
                splits=["test"],
                input_prefix="Summarize: ",
                output_prefix="Summary: "
            )
        ]
        self.max_samples = max_samples