from datetime import datetime
import json
from typing import Dict
import time
import torch.nn.functional as F
import torch
from tqdm import tqdm

# Local imports
from data.config import BenchmarkConfig
from models.base_models import ModelLoader
from data.data import DatasetManager
from data.metrics import BenchmarkMetrics
from data.config import BenchmarkConfig
from models.attention.sparse_attention import convert_attention_type
from models.h2o.h2o_llama import convert_kvcache_llama_heavy_recent
from models.attention.streaming import convert_to_streaming_attention

import gc
import os

def save_results_to_file(results: list, config: BenchmarkConfig):
    """Save benchmark results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_name.split('/')[-1]  # Get just the model name without path
    filename = f"benchmark_results_{model_name}_{timestamp}.json"
    
    try:
        with open(filename, "w") as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

# Local imports

def run_benchmark(config: BenchmarkConfig, save_results: bool = True, verbose: bool = True):
    """Run benchmarks with given configuration."""
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    results = []
    device = torch.device(config.device)
    
    try:
        # Safer memory clearing
        try:
            if torch.cuda.is_available():
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                # Safer empty cache
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            gc.collect()
        except RuntimeError as e:
            log(f"Warning: Memory cleanup error (non-critical): {e}")
        
        # Set memory allocation strategy more conservatively
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reduced from 0.9
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Load model and tokenizer
        log("\nInitializing components...")
        model_loader = ModelLoader(config)
        
        # Use mixed precision more explicitly
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.cuda.amp.autocast():
            model, tokenizer = model_loader.load_model_and_tokenizer()
            model = model.to(device).to(dtype)
            
            dataset_manager = DatasetManager(config, tokenizer)
            
            if config.attention_type != "default":
                log(f"\nConverting attention mechanism to {config.attention_type}...")
                if config.attention_type == "streaming":
                    model.config.window_size = config.window_size
                    model.config.sink_size = config.sink_size
                    model.config.sink_update_rate = config.sink_update_rate
                    model = convert_to_streaming_attention(model, model.config)
                log("Attention mechanism converted successfully")
        
        # Process datasets
        for dataset_config in config.datasets:
            for split in dataset_config.splits:
                log(f"\nTesting {dataset_config.name} ({split})")
                try:
                    dataset = dataset_manager.load_dataset(
                        dataset_config.name, 
                        split,
                        batch_size=1
                    )
                    
                    if dataset is None:
                        log(f"Skipping {dataset_config.name} - failed to load")
                        continue
                    
                    for batch in tqdm(dataset, desc="Processing examples"):
                        try:
                            if device.type == "cuda":
                                with torch.cuda.device(device):
                                    torch.cuda.empty_cache()
                            
                            with torch.cuda.amp.autocast():
                                with torch.no_grad():
                                    input_ids = batch['input_ids'].to(device)
                                    attention_mask = batch['attention_mask'].to(device)
                                    input_text = batch['text']
                                    
                                    start_time = time.time()
                                    outputs = model.generate(
                                        input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=min(config.max_tokens, 128),
                                        do_sample=True,
                                        temperature=config.temperature,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        pad_token_id=tokenizer.pad_token_id
                                    )
                                    
                                    result = process_outputs(outputs, input_text, tokenizer, start_time, config, device)
                                    results.append(result)
                                    
                            # Explicit cleanup after each batch
                            del outputs
                            del input_ids
                            del attention_mask
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                log(f"OOM error, attempting recovery: {str(e)}")
                                if torch.cuda.is_available():
                                    with torch.cuda.device(device):
                                        torch.cuda.empty_cache()
                                continue
                            raise e
                                
                except Exception as e:
                    log(f"Error testing {dataset_config.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
    finally:
        # Safer cleanup
        try:
            if 'model' in locals():
                model.cpu()
                del model
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            log(f"Warning: Final cleanup error (non-critical): {e}")
    
    if save_results and results:
        save_results_to_file(results, config)
    
    return results

def process_outputs(outputs, input_text, tokenizer, start_time, config, device):
    """Process model outputs into result format"""
    generated_tokens = outputs.sequences[0, -config.max_tokens:].detach()
    tokens = tokenizer.convert_ids_to_tokens(generated_tokens.cpu())
    
    # Calculate log probabilities
    logprobs = []
    top_logprobs = []
    for logits in outputs.scores:
        logits = logits[0].detach()  # Detach from computation graph
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        top_prob, top_idx = probs.max(dim=-1)
        logprobs.append(log_probs[top_idx].cpu().item())
        top_token = tokenizer.convert_ids_to_tokens(top_idx.cpu().item())
        top_logprobs.append({top_token: log_probs[top_idx].cpu().item()})
    
    generated_text = tokenizer.decode(generated_tokens.cpu(), skip_special_tokens=True)
    generation_time = time.time() - start_time
    
    return {
        'request': {
            'prompt': input_text,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'stop': None
        },
        'result': {
            "choices": [{
                "text": generated_text,
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": logprobs,
                    "top_logprobs": top_logprobs,
                    "text_offset": []
                },
                "finish_reason": "length"
            }],
            "request_time": {
                "batch_time": generation_time,
                "batch_size": 1
            }
        }
    }


def run_attention_strategy_comparison(config, cache_sizes=[100, 80, 60, 20, 10, 4]):
    """
    Run benchmarks comparing different attention strategies:
    - Full Attention (baseline)
    - Local Attention (simple windowing)
    - Heavy Hitter Oracle (H2O)
    - Liquid Fusion (combines sink tokens with heavy hitters)
    - Streaming LLM (attention sink mechanism)
    """
    comparison_results = {
        'full': {},
        'local': {},
        'h2o': {},
        'liquid_fusion': {},
        'streaming': {}
    }
    
    base_config = copy.deepcopy(config)
    
    for strategy in comparison_results.keys():
        print(f"\nTesting {strategy} strategy...")
        
        for cache_size in cache_sizes:
            # Skip unnecessary full cache variations
            if strategy == 'full' and cache_size != 100:
                continue
                
            print(f"Testing with {cache_size}% cache budget")
            
            # Update config for current test
            current_config = copy.deepcopy(base_config)
            current_config.kv_cache_budget = cache_size
            
            # Configure model based on strategy
            if strategy == 'h2o':
                # H2O configuration
                current_config.heavy_ratio = 0.5
                current_config.recent_ratio = 0.5
                model = convert_kvcache_llama_heavy_recent(current_config.model, current_config)
            
            elif strategy == 'liquid_fusion':
                # Liquid Fusion configuration
                current_config.sink_size = 4  # Small number of sink tokens
                current_config.heavy_ratio = 0.3  # 30% for heavy hitters
                current_config.recent_ratio = 0.3  # 30% for recent tokens
                current_config.sink_update_rate = 0.1  # EMA rate for sink updates
                model = convert_to_liquid_fusion(current_config.model, current_config)
            
            elif strategy == 'streaming':
                # StreamingLLM configuration
                current_config.window_size = int(current_config.model.config.max_position_embeddings * (cache_size/100))
                current_config.sink_size = min(32, current_config.window_size // 8)
                model = convert_to_streaming_attention(current_config.model, current_config)
            
            elif strategy == 'local':
                # Local attention (only recent tokens)
                current_config.recent_ratio = 1.0
                current_config.heavy_ratio = 0.0
                model = convert_kvcache_llama_heavy_recent(current_config.model, current_config)
            
            # Run benchmark
            try:
                results = run_benchmark(current_config, save_results=False, verbose=False)
                
                # Calculate metrics
                rouge_scores = calculate_rouge_scores(results, current_config.reference_texts)
                throughput = calculate_throughput(results)
                memory_usage = measure_memory_usage()
                
                comparison_results[strategy][cache_size] = {
                    **rouge_scores,
                    'throughput': throughput,
                    'memory': memory_usage
                }
                
            except Exception as e:
                print(f"Error testing {strategy} with {cache_size}% cache: {str(e)}")
                continue
            
            # Clear memory
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            gc.collect()
    
    return comparison_results

def plot_comparison_metrics(results, metric='rouge1', title=None):
    """Create visualizations comparing different strategies across multiple metrics."""
    plt.figure(figsize=(12, 7))
    
    # Define consistent colors and markers for each strategy
    style_dict = {
        'full': ('blue', 'o', 'Full'),
        'local': ('red', 's', 'Local'),
        'h2o': ('green', '^', 'H2O'),
        'liquid_fusion': ('purple', 'D', 'Liquid Fusion'),
        'streaming': ('orange', 'v', 'StreamingLLM')
    }
    
    for strategy, measurements in results.items():
        cache_sizes = sorted(measurements.keys())
        scores = [measurements[size][metric] for size in cache_sizes]
        
        color, marker, label = style_dict[strategy]
        plt.plot(cache_sizes, scores, 
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=1)
    
    plt.xlabel('KV Cache Budget (%)')
    plt.ylabel(f'{metric.upper()} Score')
    plt.title(title or f'{metric.upper()} vs KV Cache Budget')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt

# Helper functions for additional metrics
def calculate_throughput(results):
    """Calculate average tokens/second"""
    total_time = sum(r['result']['request_time']['batch_time'] for r in results)
    total_tokens = sum(len(r['result']['choices'][0]['logprobs']['tokens']) for r in results)
    return total_tokens / total_time

def measure_memory_usage():
    """Measure GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    return 0
