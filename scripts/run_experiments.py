import os
import argparse
import logging
import numpy as np
import torch
import json
import tqdm
import copy
from datetime import datetime

from data.config import BenchmarkConfig, DatasetConfig
from models.base_models import ModelLoader
from data.data import DatasetManager
from scripts.run_benchmark import run_benchmark, run_single_strategy_benchmark

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def get_device():
    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    logger.info(f"Using device: {device}")
    return device

def get_base_config(args):
    return BenchmarkConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        device=args.device,
        sequence_length=args.sequence_length,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        datasets=[
            DatasetConfig(
                name="super_glue",
                config="copa",
                splits=["test"],
                input_prefix="Question: ",
                output_prefix="Answer: ",
                max_samples=args.max_samples
            )
        ]
    )

def run_strategy_tests(base_config, strategies, cache_sizes):
    results = {}
    
    for strategy in strategies:
        strategy_config = copy.deepcopy(base_config)
        
        if strategy == "full":
            strategy_config.attention_type = "default"
        elif strategy == "h2o":
            strategy_config.attention_type = "h2o"
            strategy_config.heavy_ratio = 0.1
            strategy_config.recent_ratio = 0.1
        elif strategy == "streaming":
            strategy_config.attention_type = "streaming"
            strategy_config.window_size = 64
            strategy_config.sink_size = 4
            strategy_config.sink_update_rate = 0.1
        elif strategy == "local":
            strategy_config.attention_type = "local"
            strategy_config.window_size = 64
        elif strategy == "liquid_fusion":
            strategy_config.attention_type = "liquid_fusion"
            strategy_config.window_size = 64
            strategy_config.sink_size = 2
            strategy_config.sink_update_rate = 0.1
            strategy_config.heavy_ratio = 0.1
            strategy_config.recent_ratio = 0.1
            
        for cache_size in cache_sizes:
            logger.info(f"Testing {strategy} with {cache_size}% cache")
            result = run_single_strategy_benchmark(strategy_config, strategy=strategy, cache_size=cache_size)
            results[f"{strategy}_{cache_size}"] = result
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Run attention mechanism experiments")
    parser.add_argument("--model-name", default="huggyllama/llama-7b", help="Model name or path")
    parser.add_argument("--model-type", default="llama", help="Model type")
    parser.add_argument("--device", default=get_device(), help="Device to run on")
    parser.add_argument("--sequence-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to test")
    parser.add_argument("--mini-test", action="store_true", help="Run mini test with 5 samples")
    parser.add_argument("--strategies", nargs="+", 
                       default=["full", "h2o", "streaming", "local", "liquid_fusion"],
                       help="Strategies to test")
    parser.add_argument("--cache-sizes", nargs="+", type=int,
                       default=[100, 80, 40, 20, 4],
                       help="Cache sizes to test")
    
    args = parser.parse_args()
    setup_logging()
    
    if args.mini_test:
        args.max_samples = 5
        args.cache_sizes = [100, 20]  # Reduced cache sizes for mini test
        
    base_config = get_base_config(args)
    results = run_strategy_tests(base_config, args.strategies, args.cache_sizes)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()