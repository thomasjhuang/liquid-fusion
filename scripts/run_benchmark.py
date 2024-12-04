from datetime import datetime
import json
from typing import Dict
import time
import torch.nn.functional as F
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer

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
import copy
import psutil
from collections import defaultdict

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
        # Check memory at start
        detailed_memory_check("Before Benchmark")
        
        # Aggressive cleanup before starting
        aggressive_memory_cleanup()
        
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
        # Cleanup at end
        aggressive_memory_cleanup(model if 'model' in locals() else None)
    
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
    
    # Load model once and reuse
    model_loader = ModelLoader(base_config)
    base_model, tokenizer = model_loader.load_model_and_tokenizer()
    
    try:
        for strategy in comparison_results.keys():
            print(f"\nTesting {strategy} strategy...")
            deep_memory_cleanup()  # Clean between strategies
            
            for cache_size in cache_sizes:
                if strategy == 'full' and cache_size != 100:
                    continue
                    
                print(f"Testing with {cache_size}% cache budget")
                model = copy.deepcopy(base_model)
                
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
                    
                finally:
                    deep_memory_cleanup(model)  # Ensure cleanup happens
    finally:
        deep_memory_cleanup(base_model)  # Final cleanup
    
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

def calculate_rouge_scores(results, reference_texts=None):
    """Calculate ROUGE scores for generated texts"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    for result in results:
        generated = result['result']['choices'][0]['text']
        reference = result['request']['prompt'].split('Summary: ')[-1]
        
        # Calculate scores
        score = scorer.score(reference, generated)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure
    
    # Average scores
    n = len(results)
    return {k: v/n for k, v in scores.items()}

def detailed_memory_check(label=""):
    """Detailed memory analysis with tensor tracking"""
    print(f"\n=== DETAILED MEMORY ANALYSIS: {label} ===")
    
    # 1. CUDA Memory
    if torch.cuda.is_available():
        print("\nCUDA Memory Breakdown:")
        device = torch.cuda.current_device()
        
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
        
        # Get memory stats for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory used: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    
    # 2. Tensor Analysis
    print("\nTensor Memory Analysis:")
    tensor_count = defaultdict(int)
    tensor_memory = defaultdict(int)
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                key = f"{obj.device}_{obj.dtype}"
                tensor_count[key] += 1
                tensor_memory[key] += obj.element_size() * obj.nelement()
                
                # Print large tensors (>100MB)
                if obj.element_size() * obj.nelement() > 100 * 1024 * 1024:
                    print(f"\nLarge tensor found:")
                    print(f"Size: {obj.element_size() * obj.nelement() / 1024**3:.2f} GB")
                    print(f"Shape: {obj.shape}")
                    print(f"Device: {obj.device}")
                    print(f"Dtype: {obj.dtype}")
        except Exception:
            continue
    
    print("\nTensor Summary:")
    for key in tensor_count:
        print(f"{key}:")
        print(f"  Count: {tensor_count[key]}")
        print(f"  Memory: {tensor_memory[key] / 1024**3:.2f} GB")
    
    # 3. System Memory
    process = psutil.Process()
    print(f"\nSystem Memory:")
    print(f"RSS: {process.memory_info().rss / 1024**3:.2f} GB")
    print(f"VMS: {process.memory_info().vms / 1024**3:.2f} GB")

def aggressive_memory_cleanup(model=None):
    """Aggressively clean up memory"""
    print("\nStarting aggressive memory cleanup...")
    
    detailed_memory_check("Before Cleanup")
    
    # 1. Clear model
    if model is not None:
        try:
            model.cpu()
            del model
        except:
            print("Error clearing model")
    
    # 2. Clear CUDA cache multiple times
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"CUDA cleanup error: {e}")
    
    # 3. Clear all tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    obj.cpu()
                del obj
        except:
            continue
    
    # 4. Multiple GC runs
    for _ in range(3):
        gc.collect()
    
    # 5. Clear CUDA cache again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    detailed_memory_check("After Cleanup")
    
    print("\nMemory cleanup completed")

def deep_memory_cleanup(model=None, verbose=True):
    """Comprehensive memory cleanup including model weights, CUDA cache, and system memory."""
    if verbose:
        print("\n=== Starting Deep Memory Cleanup ===")
        
        # Initial memory check
        if torch.cuda.is_available():
            print("\nInitial CUDA Memory:")
            device = torch.cuda.current_device()
            print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
            print(f"Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    
    # 1. Clear model and its weights if provided
    if model is not None:
        try:
            # Move model to CPU and clear its memory
            model.cpu()
            for param in model.parameters():
                if hasattr(param, 'data'):
                    param.data = None
                if hasattr(param, 'grad'):
                    param.grad = None
            
            # Clear buffers and caches
            for buffer in model.buffers():
                if hasattr(buffer, 'data'):
                    buffer.data = None
            
            if hasattr(model, 'past_key_values'):
                model.past_key_values = None
            
            # Clear state dict and delete model
            model.state_dict().clear()
            del model
        except Exception as e:
            if verbose:
                print(f"Warning during model clearing: {e}")
    
    # 2. Clear all remaining tensors and CUDA storage
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    obj.cpu()
                obj.data = None
                del obj
            elif hasattr(obj, 'storage') and hasattr(obj.storage, '_cuda_storage'):
                obj.storage._cuda_storage = None
        except:
            continue
    
    # 3. Clear CUDA memory if available
    if torch.cuda.is_available():
        try:
            # Multiple passes of cache clearing for each device
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    for _ in range(3):
                        torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            
            # Reset memory allocator
            if hasattr(torch.cuda.memory, '_reset_memory_allocator'):
                torch.cuda.memory._reset_memory_allocator()
        except Exception as e:
            if verbose:
                print(f"Error during CUDA cleanup: {e}")
    
    # 4. Multiple GC runs
    for _ in range(5):
        gc.collect()
    
    # 5. Final memory check if verbose
    if verbose:
        print("\nFinal Memory Status:")
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
        # System memory
        process = psutil.Process()
        print(f"System RSS: {process.memory_info().rss / 1024**3:.2f} GB")
        print("\n=== Memory Cleanup Completed ===")
    
    return None
