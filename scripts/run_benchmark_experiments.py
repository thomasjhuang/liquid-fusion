import torch
# Import necessary libraries
import argparse
import json
import torch
from tqdm import tqdm
import logging
from datetime import datetime
import gc
from transformers import AutoTokenizer, GenerationConfig
from models.base_models import ModelLoader
from models.attention.sparse_attention import convert_attention_type
from data.config import BenchmarkConfig, DatasetConfig
import time, copy
from datasets import load_dataset
from rouge_score import rouge_scorer
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display, clear_output
from scripts.run_benchmark import run_benchmark

# Set up device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

streaming_config = BenchmarkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    model_type="llama",
    device=device,
    dtype="float16",
    strategies=["streaming"],
    max_new_tokens=128,
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
    model_name="meta-llama/Llama-2-7b-chat-hf",
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
    model_name="meta-llama/Llama-2-7b-chat-hf",
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
            input_prefix="Summarize this article:\n\n",
            output_prefix="\n\nSummary:",
            max_samples=10
        )
    ]
)


# Base configurations
def get_base_config(strategy):
    return BenchmarkConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_type="llama",
        device=device,
        dtype="float16",
        strategies=[strategy],
        max_new_tokens=128,
        sequence_length=4096,
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

# Function to run benchmark and store results
def run_experiment(config, experiment_name):
    results = run_benchmark(config)
    
    # Store results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    results_path = results_dir / f"{experiment_name}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, results_path

# Visualization functions
def plot_sequence_length_comparison(results_dict):
    """Plot comparison of methods across sequence lengths"""
    metrics = ['avg_rouge1', 'avg_rouge2', 'avg_rougeL'] 
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, metric in enumerate(metrics):
        for method in results_dict.keys():
            # Get sequence lengths where there isn't an error
            seq_lengths = []
            values = []
            
            for seq_len in results_dict[method].keys():
                # Check if there are results (not an error)
                if method in results_dict[method][seq_len] and isinstance(results_dict[method][seq_len][method], dict) and 'error' not in results_dict[method][seq_len][method]:
                    seq_lengths.append(seq_len)
                    values.append(results_dict[method][seq_len][method][metric])
            
            if seq_lengths and values:
                axes[idx].plot(seq_lengths, values, marker='o', label=method)
        
        axes[idx].set_title(f'{metric} vs Sequence Length')
        axes[idx].set_xlabel('Sequence Length')
        axes[idx].set_ylabel(metric)
        axes[idx].legend()
        axes[idx].grid(True)
    
    plt.tight_layout()
    return fig

def plot_parameter_sensitivity(results_dict, param_name):
    metrics = ['avg_rouge1', 'avg_inference_time', 'avg_tokens_per_second']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, metric in enumerate(metrics):
        param_values = list(results_dict.keys())
        values = [results_dict[param][metric] for param in param_values]
        axes[idx].plot(param_values, values, marker='o')
        
        axes[idx].set_title(f'{metric} vs {param_name}')
        axes[idx].set_xlabel(param_name)
        axes[idx].set_ylabel(metric)
        axes[idx].grid(True)
    
    plt.tight_layout()
    return fig

# Experiment 1: Sequence Length Comparison
def run_sequence_length_experiment():
    sequence_lengths = [512, 1024, 2048, 4096]
    results_by_length = {
        'streaming': {},
        'h2o': {},
        'liquid_fusion': {}
    }
    
    for seq_len in tqdm(sequence_lengths, desc="Testing sequence lengths"):
        # StreamingLLM config
        streaming_config = get_base_config("streaming")
        streaming_config.sequence_length = seq_len
        streaming_config.start_size = 4
        streaming_config.recent_size = min(int(0.85 * seq_len), seq_len - 4)  # Ensure we don't exceed sequence length
        streaming_config.max_length = seq_len  # Add this line

        # H2O config
        h2o_config = get_base_config("h2o")
        h2o_config.sequence_length = seq_len
        h2o_config.heavy_budget = int(0.2 * seq_len)
        h2o_config.recent_budget = int(0.2 * seq_len)
        
        # Liquid Fusion config
        liquid_config = get_base_config("liquid_fusion")
        liquid_config.sequence_length = seq_len
        liquid_config.heavy_budget = int(0.2 * seq_len)
        liquid_config.recent_budget = int(0.2 * seq_len)
        
        # Run benchmarks
        results_by_length['streaming'][seq_len] = run_benchmark(streaming_config)
        results_by_length['h2o'][seq_len] = run_benchmark(h2o_config)
        results_by_length['liquid_fusion'][seq_len] = run_benchmark(liquid_config)
        
        # Clear cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return results_by_length

# Save all results
def save_all_results(results_dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    results_path = results_dir / f"all_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_path

def get_gpu_memory_stats():
    """Get detailed GPU memory statistics"""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'cached': 0, 'peak': 0}
        
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # Currently allocated 
        'cached': torch.cuda.memory_reserved() / 1024**3,      # Cached by PyTorch allocator
        'peak': torch.cuda.max_memory_allocated() / 1024**3    # Peak allocation
    }

def run_memory_experiment():
    """Run benchmark comparing memory usage across all attention mechanisms"""
    base_seq_length = 4096
    cache_sizes = [512, 1024, 2048, 4096]
    results = {
        'full': {},        
        'streaming': {},
        'h2o': {},
        'liquid_fusion': {}
    }
    
    for cache_size in tqdm(cache_sizes, desc="Testing cache sizes"):
        print(f"\nTesting cache size: {cache_size}")
        
        # Create base configs for each method
        base_config = get_base_config("full")  
        streaming_config = get_base_config("streaming")
        h2o_config = get_base_config("h2o")
        liquid_config = get_base_config("liquid_fusion")
        
        configs = {
            'full': base_config,
            'streaming': streaming_config,
            'h2o': h2o_config,
            'liquid_fusion': liquid_config
        }
        
        # Update sequence lengths and cache sizes for each config
        for method, config in configs.items():
            config.sequence_length = cache_size
            
            if method == 'streaming':
                config.start_size = 4
                config.recent_size = cache_size - 4
            elif method in ['h2o', 'liquid_fusion']:
                config.heavy_budget = cache_size // 2
                config.recent_budget = cache_size // 2
            # Full attention doesn't need special parameters
        
        for method, config in configs.items():
            try:
                # Clear cache and track initial memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                initial_stats = get_gpu_memory_stats()
                
                # Run benchmark
                results_dict = run_benchmark(config)
                
                # Get final memory stats
                final_stats = get_gpu_memory_stats()
                
                # Store results with memory metrics
                results[method][cache_size] = {
                    method: {
                        **results_dict[method],
                        'initial_memory': initial_stats,
                        'final_memory': final_stats,
                        'peak_memory': final_stats['peak']
                    }
                }
                
            except Exception as e:
                print(f"Error in {method}: {str(e)}")
                results[method][cache_size] = {method: {'error': str(e)}}
                
            # Clean up after each run
            torch.cuda.empty_cache()
            gc.collect()
            
            # Add a small delay between runs to ensure clean state
            time.sleep(2)
    
    return results

def plot_detailed_memory_comparison(results_dict):
    """Enhanced plotting function with full attention included"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors for better visualization
    colors = {
        'full': 'red',
        'streaming': 'blue',
        'h2o': 'green',
        'liquid_fusion': 'purple'
    }
    
    # Plot cache size vs peak memory
    for method in results_dict:
        sizes = []
        peaks = []
        for size in results_dict[method]:
            if 'error' not in results_dict[method][size][method]:
                sizes.append(size)
                peaks.append(results_dict[method][size][method]['peak_memory'])
        ax1.plot(sizes, peaks, marker='o', label=method, color=colors[method])
    ax1.set_title('Peak Memory Usage')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Memory (GB)')
    ax1.grid(True)
    ax1.legend()

    # Plot cache size vs inference time
    for method in results_dict:
        sizes = []
        times = []
        for size in results_dict[method]:
            if 'error' not in results_dict[method][size][method]:
                sizes.append(size)
                times.append(results_dict[method][size][method]['avg_inference_time'])
        ax2.plot(sizes, times, marker='o', label=method, color=colors[method])
    ax2.set_title('Average Inference Time')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Time (s)')
    ax2.grid(True)
    ax2.legend()

    # Plot cache size vs ROUGE-1
    for method in results_dict:
        sizes = []
        scores = []
        for size in results_dict[method]:
            if 'error' not in results_dict[method][size][method]:
                sizes.append(size)
                scores.append(results_dict[method][size][method]['avg_rouge1'])
        ax3.plot(sizes, scores, marker='o', label=method, color=colors[method])
    ax3.set_title('ROUGE-1 Score')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend()

    # Plot cache size vs throughput
    for method in results_dict:
        sizes = []
        tps = []
        for size in results_dict[method]:
            if 'error' not in results_dict[method][size][method]:
                sizes.append(size)
                tps.append(results_dict[method][size][method]['avg_tokens_per_second'])
        ax4.plot(sizes, tps, marker='o', label=method, color=colors[method])
    ax4.set_title('Throughput')
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Tokens/second')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    return fig
