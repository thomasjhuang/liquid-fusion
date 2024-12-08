from datetime import datetime
import json
from typing import Dict
import time
import torch.nn.functional as F
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import math
import numpy as np
import logging


# Local imports
from data.config import BenchmarkConfig
from models.base_models import ModelLoader
from data.data import DatasetManager
from data.metrics import BenchmarkMetrics
from data.config import BenchmarkConfig
from models.attention.sparse_attention import convert_attention_type
from models.h2o.h2o_llama import convert_kvcache_llama_heavy_recent
from models.attention.streaming import convert_to_streaming_attention
from models.attention.liquid_fusion import convert_to_liquid_fusion

import gc
import os
import copy
import psutil
from collections import defaultdict

# Initialize logger
logger = logging.getLogger(__name__)

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

def run_benchmark(config: BenchmarkConfig, save_results: bool = True, verbose: bool = True):
    """Run benchmarks with given configuration."""
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    results = []
    device = torch.device(config.device)
    
    try:
        # Simple cleanup before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Allow for more aggressive memory usage
            torch.cuda.set_per_process_memory_fraction(0.95)  # Increased from 0.8
            # Configure PyTorch memory allocator
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        gc.collect()
        
        # Load model and tokenizer
        log("\nInitializing components...")
        model_loader = ModelLoader(config)
        
        with torch.cuda.amp.autocast():
            model, tokenizer = model_loader.load_model_and_tokenizer()
            model = model.to(device)
            
            dataset_manager = DatasetManager(config, tokenizer)
            
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
                                        
                                        result = process_outputs(outputs, input_text, batch['reference_text'], tokenizer, start_time, config, device)
                                        results.append(result)
                                
                                # Cleanup after each batch
                                del outputs, input_ids, attention_mask
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    log(f"OOM error, attempting recovery")
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                                raise e
                                    
                    except Exception as e:
                        log(f"Error testing {dataset_config.name}: {str(e)}")
                        continue
                    
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    if save_results and results:
        save_results_to_file(results, config)
    
    return results

def process_outputs(outputs, input_text, reference_text, tokenizer, start_time, config, device):
    """Process model outputs into result format with comprehensive metrics for strategy comparison"""
    generated_tokens = outputs.sequences[0, -config.max_tokens:].detach()
    generated_text = tokenizer.decode(generated_tokens.cpu(), skip_special_tokens=True)
    generation_time = time.time() - start_time
    
    # Calculate perplexity
    logprobs = []
    for logits in outputs.scores:
        logits = logits[0].detach()
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        top_prob, top_idx = probs.max(dim=-1)
        logprobs.append(log_probs[top_idx].cpu().item())
    
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0
    perplexity = math.exp(-mean_logprob) if mean_logprob != 0 else float('inf')
    
    # Memory metrics
    memory_metrics = {
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0,
        "current_memory_mb": torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0,
        "memory_utilization": torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0,
    }
    
    # Cache metrics
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
        cache_size = sum(sum(x[0].nelement() * x[0].element_size() + 
                           x[1].nelement() * x[1].element_size() 
                           for x in layer) 
                        for layer in outputs.past_key_values) / 1024**2
        cache_length = outputs.past_key_values[0][0].shape[2]
    else:
        cache_size = 0
        cache_length = 0 
        
    # Performance metrics
    tokens_per_second = config.max_tokens / generation_time if generation_time > 0 else 0
    
    # Quality metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, generated_text)
    exact_match_score = 1 if reference_text.strip() == generated_text.strip() else 0
    
    return {
        'request': {
            'prompt': input_text,
            'reference': reference_text,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
        },
        'result': {
            "choices": [{
                "text": generated_text,
                "logprobs": {
                    "token_logprobs": logprobs,
                },
                "finish_reason": "length"
            }],
            "metrics": {
                # Quality metrics
                "perplexity": perplexity,
                "exact_match": exact_match_score,
                "rouge1_f": rouge_scores['rouge1'].fmeasure,
                "rouge2_f": rouge_scores['rouge2'].fmeasure,
                "rougeL_f": rouge_scores['rougeL'].fmeasure,
                
                # Performance metrics
                "generation_time_ms": generation_time * 1000,
                "tokens_per_second": tokens_per_second,
                
                # Memory metrics
                **memory_metrics,
                
                # Cache metrics
                "cache_size_mb": cache_size,
                "cache_length": cache_length,
            }
        }
    }

def save_benchmark_results(results, config, strategy, cache_size):
    """Save benchmark results with detailed configuration info."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_name.split('/')[-1]
    
    # Create results directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Create detailed filename
    filename = f"benchmark_results/{strategy}_{model_name}_cache{cache_size}_{timestamp}.json"
    
    def convert_tensors(obj):
        """Recursively convert tensors to Python types"""
        if torch.is_tensor(obj):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {key: convert_tensors(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_tensors(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            return convert_tensors(obj.__dict__)
        return obj
    
    # Prepare metadata
    metadata = {
        "model_name": config.model_name,
        "strategy": strategy,
        "cache_size": cache_size,
        "attention_type": config.attention_type,
        "window_size": getattr(config, 'window_size', None),
        "sink_size": getattr(config, 'sink_size', None),
        "sink_update_rate": getattr(config, 'sink_update_rate', None),
        "heavy_ratio": getattr(config, 'heavy_ratio', None),
        "recent_ratio": getattr(config, 'recent_ratio', None),
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "timestamp": timestamp,
        "device": str(config.device)
    }
    
    # Convert results to JSON-serializable format
    serializable_results = convert_tensors(results)
    
    # Combine metadata with results
    full_results = {
        "metadata": metadata,
        "results": serializable_results
    }
    
    try:
        with open(filename, "w") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"\nResults saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        # Print the problematic data for debugging
        logger.debug("Metadata:", metadata)
        logger.debug("First result entry:", next(iter(serializable_results.items())) if isinstance(serializable_results, dict) else serializable_results[0] if serializable_results else None)

def run_single_strategy_benchmark(config, strategy, cache_size):
    """Run benchmark with fast generation and deferred metrics computation"""
    logger.info(f"\nTesting {strategy} strategy with {cache_size}% cache")
    
    results = {
        "metadata": {
            "model_name": config.model_name,
            "strategy": strategy,
            "cache_size": cache_size,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "generations": []
    }
    
    try:
        # Setup
        logger.info("Cleaning up memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_model_and_tokenizer()
        dataset_manager = DatasetManager(config, tokenizer)
        dataloader = dataset_manager.get_dataloader()
        device = next(model.parameters()).device
        
        # Get max samples from config or dataset size
        num_examples = config.datasets[0].max_samples if config.datasets and config.datasets[0].max_samples else len(dataloader)
        
        # Fast generation loop
        for i, batch in enumerate(tqdm(dataloader, desc="Generating outputs")):
            if i >= num_examples:
                break
                
            # Minimal processing for generation
            with torch.no_grad():
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                }
                
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                )
                generation_time = time.time() - start_time
                
                # Store minimal data needed for later processing
                results["generations"].append({
                    "input_text": batch.get('text', tokenizer.decode(batch['input_ids'][0])),
                    "generated_ids": outputs.sequences[0].cpu().tolist(),
                    "reference_text": batch.get('reference_text', batch.get('target', batch.get('label', ""))),
                    "latency": generation_time
                })
                
                # Immediate cleanup
                del outputs
                torch.cuda.empty_cache()
        
        # Compute metrics after all generations (optional)
        if config.compute_metrics:
            logger.info("Computing metrics...")
            metrics = compute_metrics_from_generations(results["generations"], tokenizer)
            results["metrics"] = metrics
        
        # Save results
        save_benchmark_results(results, config, strategy, cache_size)
        return results
        
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}", exc_info=True)
        return results
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def compute_metrics_from_generations(generations, tokenizer):
    """Compute metrics after all generations are complete"""
    metrics = {
        'per_example': [],
        'summary': {}
    }
    
    # Decode all generations at once
    generated_texts = [tokenizer.decode(gen['generated_ids'], skip_special_tokens=True) 
                      for gen in generations]
    
    # Process all examples
    for gen, decoded_text in zip(generations, generated_texts):
        example_metrics = {
            'latency_ms': gen['latency'] * 1000,
            'input_text': gen['input_text'],
            'generated_text': decoded_text,
            'reference_text': gen['reference_text']
        }
        metrics['per_example'].append(example_metrics)
    
    # Batch compute ROUGE scores
    if any(m['reference_text'] for m in metrics['per_example']):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [
            scorer.score(str(m['reference_text']), str(m['generated_text']))
            for m in metrics['per_example']
        ]
        
        for i, scores in enumerate(rouge_scores):
            metrics['per_example'][i].update({
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure
            })
    
    # Compute summary statistics
    metrics['summary'] = {
        'avg_latency_ms': np.mean([m['latency_ms'] for m in metrics['per_example']]),
        'avg_tokens_per_second': np.mean([len(m['generated_text'].split()) / (m['latency_ms'] / 1000) 
                                        for m in metrics['per_example']])
    }
    
    if 'rouge1_f' in metrics['per_example'][0]:
        metrics['summary'].update({
            'avg_rouge1_f': np.mean([m['rouge1_f'] for m in metrics['per_example']]),
            'avg_rouge2_f': np.mean([m['rouge2_f'] for m in metrics['per_example']]),
            'avg_rougeL_f': np.mean([m['rougeL_f'] for m in metrics['per_example']])
        })
    
    return metrics

def calculate_summary_metrics(metrics):
    """Calculate comprehensive summary statistics"""
    if not metrics:
        return {}
    
    def safe_mean(key):
        values = [m[key] for m in metrics if key in m and m[key] is not None]
        return sum(values) / len(values) if values else None
    
    summary = {
        "total_samples": len(metrics),
        "avg_input_length": safe_mean("input_length"),
        "avg_output_length": safe_mean("output_length"),
        "avg_latency_ms": safe_mean("latency_ms"),
        "avg_tokens_per_second": safe_mean("tokens_per_second"),
        "avg_perplexity": safe_mean("perplexity"),
    }
    
    # Quality metrics
    if any('rouge1_fmeasure' in m for m in metrics):
        summary.update({
            "avg_rouge1_f1": safe_mean("rouge1_fmeasure"),
            "avg_rouge2_f1": safe_mean("rouge2_fmeasure"),
            "avg_rougeL_f1": safe_mean("rougeL_fmeasure"),
            "exact_match_rate": safe_mean("exact_match")
        })
    
    # Memory metrics
    if any('kv_cache_size_mb' in m for m in metrics):
        summary.update({
            "avg_kv_cache_size_mb": safe_mean("kv_cache_size_mb"),
            "max_kv_cache_size_mb": max(m["kv_cache_size_mb"] for m in metrics if "kv_cache_size_mb" in m),
        })
    
    return summary

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

def deep_memory_cleanup(model=None, verbose=False):
    """Comprehensive memory cleanup including model weights, CUDA cache, and system memory."""
    print("Cleaning up memory...")
    
    if model is not None:
        try:
            model.cpu()
            for param in model.parameters():
                if hasattr(param, 'data'):
                    param.data = None
                if hasattr(param, 'grad'):
                    param.grad = None
            model.state_dict().clear()
            del model
        except Exception:
            pass
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    
    for _ in range(5):
        gc.collect()
