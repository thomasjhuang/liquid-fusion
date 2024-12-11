from data.config import BenchmarkConfig, DatasetConfig
from data.data import DatasetManager, ModelDataset
from data.metrics import BenchmarkMetrics
from models.base_models import ModelLoader
from scripts.run_benchmark import run_single_strategy_benchmark, run_benchmark
import copy

import json
from typing import List, Optional, Tuple, Dict
import time
import torch, transformers
from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from itertools import zip_longest

print(f"torch version: {torch.__version__}")
print(f"transformers version: {transformers.__version__}")

device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")

streaming_config = BenchmarkConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
    model_type="llama",
    device=device,
    dtype="float16",
    strategies=["streaming"],
    max_new_tokens=256,
    sequence_length=4096,
    start_size=4,
    recent_size=3496,
    max_samples=10,
    datasets=[
        DatasetConfig(
            name="csebuetnlp/xlsum",
            splits=["test"],
            config='english',
            input_prefix="Summarize this article:\n\n",
            output_prefix="\n\nSummary:",
            max_samples=10
        )
    ]
)

h2o_config = BenchmarkConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
    model_type="llama",
    device=device,
    dtype="float16",
    strategies=["h2o"],
    max_new_tokens=128,
    sequence_length=4096,
    heavy_budget=800,
    recent_budget=800,
    max_samples=10,
    datasets=[
        DatasetConfig(
            name="csebuetnlp/xlsum",
            splits=["test"],
            config='en',
            input_prefix="Summarize this article:\n\n",
            output_prefix="\n\nSummary:",
            max_samples=10
        )
    ]
)

liquid_config = BenchmarkConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
    model_type="llama",
    device=device,
    dtype="float16",
    strategies=["liquid_fusion"],
    max_new_tokens=128,
    sequence_length=4096,
    heavy_budget=800,
    recent_budget=800,
    max_samples=10,
    datasets=[
        DatasetConfig(
            name="csebuetnlp/xlsum",
            splits=["test"],
            config='en',
            input_prefix="Summarize this article:\n\n",
            output_prefix="\n\nSummary:",
            max_samples=10
        )
    ]
)

# run_benchmark(h2o_config)
run_benchmark(streaming_config)
# run_benchmark(liquid_config)
