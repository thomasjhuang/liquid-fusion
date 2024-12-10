import argparse
import json
import torch
from tqdm import tqdm
import logging
from datetime import datetime
import gc
from transformers import AutoTokenizer
from models.base_models import ModelLoader
from models.attention.sparse_attention import convert_attention_type
from data.config import BenchmarkConfig
import time, copy
from models.streaming_llm.enable_streaming_llm import enable_streaming_llm
from datasets import load_dataset

logger = logging.getLogger(__name__)

def log_cache_stats(model, past_key_values=None):
    """Log cache statistics for models"""
    
    cache_metrics = CacheMetrics()
    cache_metrics.reset()  # Reset for new measurement
    
    if past_key_values is not None:
        cache_metrics.update_from_past_key_values(past_key_values)
        stats = cache_metrics.get_stats()
        logger.info(f"KV Cache Stats: Memory={stats['total_memory_mb']:.2f}MB, "
                   f"Avg Tokens={stats['avg_tokens_cached']:.1f}, "
                   f"Layers={stats['num_layers']}")
        return stats
    return {}

def get_generation_config(tokenizer, input_ids):
    """Standardized generation configuration with proper attention mask"""
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones_like(input_ids)
    
    # Ensure tokenizer has proper padding settings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,  # Add attention mask
        'max_new_tokens': 32,
        'do_sample': False,
        'num_return_sequences': 1,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
        'return_dict_in_generate': True
    }

def run_single_strategy_benchmark(config, strategy):
    """Run benchmark for a single attention strategy"""
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    # Configure tokenizer properly
    tokenizer.padding_side = "left"  # Typically better for casual LM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if strategy == "streaming":
        enable_streaming_llm(
            model,
            start_size=config.start_size,
            recent_size=config.recent_size
        )
        logger.info(f"Using StreamingLLM with start_size={config.start_size}, recent_size={config.recent_size}")
    elif strategy == "h2o":
        config.heavy_ratio = getattr(config, 'heavy_ratio', 0.1)
        config.recent_ratio = getattr(config, 'recent_ratio', 0.1)
        logger.info(f"Using H2O attention with total budget={config.heavy_ratio + config.recent_ratio:.1f}")
    elif strategy == "liquid_fusion":
        config.start_size = getattr(config, 'start_size', 4)
        config.recent_size = getattr(config, 'recent_size', 64)
        config.heavy_ratio = getattr(config, 'heavy_ratio', 0.1)
        config.recent_ratio = getattr(config, 'recent_ratio', 0.1)
        logger.info(f"Using LiquidFusion with start_size={config.start_size}, "
                   f"recent_size={config.recent_size}, "
                   f"heavy_ratio={config.heavy_ratio}, "
                   f"recent_ratio={config.recent_ratio}")
    
    # Load COPA dataset
    dataset = load_dataset("super_glue", "copa", split="validation")
    
    if config.max_samples:
        dataset = dataset.select(range(config.max_samples))

    results = []
    metrics = {
        'correct': 0,
        'total': 0,
        'accuracy': 0.0,
        'memory_usage': [],
        'inference_times': [],
        'cache_stats': []
    }
    
    # Initialize cache metrics
    # from data.metrics import CacheMetrics
    # cache_metrics = CacheMetrics()
    # cache_metrics.reset()
    
    with torch.no_grad():
        for example in tqdm(dataset, desc=f"Evaluating {strategy}"):
            # Device-specific timing setup
            if config.device == "cuda" and torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            # Format COPA prompt
            premise = example['premise']
            choice1 = example['choice1']
            choice2 = example['choice2']
            question = "cause" if example['question'] == 0 else "effect"
            
            prompt = f"Given the {question}: '{premise}'\nWhich is more likely?\n1. {choice1}\n2. {choice2}\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            # Track memory based on device
            if config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
            elif config.device == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
                start_mem = torch.mps.current_allocated_memory()
            
            # Generate response
            generation_config = get_generation_config(tokenizer, input_ids)
            output = model.generate(**generation_config)
            
            # Get cache info from the model's last forward pass
            if hasattr(model, 'last_forward_cache'):
                cache_metrics.update_from_past_key_values(model.last_forward_cache)
            elif hasattr(output, 'past_key_values'):
                cache_metrics.update_from_past_key_values(output.past_key_values)
            
            # Record timing based on device
            if config.device == "cuda" and torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            
            metrics['inference_times'].append(elapsed_time)
            
            # Record memory based on device
            if config.device == "cuda" and torch.cuda.is_available():
                end_mem = torch.cuda.memory_allocated()
                metrics['memory_usage'].append(end_mem - start_mem)
            elif config.device == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
                end_mem = torch.mps.current_allocated_memory()
                metrics['memory_usage'].append(end_mem - start_mem)
            
            generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Process answer
            predicted_choice = 0
            if "1" in generated_text[:10]:
                predicted_choice = 0
            elif "2" in generated_text[:10]:
                predicted_choice = 1
                
            is_correct = predicted_choice == example['label']
            metrics['correct'] += int(is_correct)
            metrics['total'] += 1
            
            
            results.append({
                'premise': premise,
                'question': question,
                'choice1': choice1,
                'choice2': choice2,
                'predicted': predicted_choice,
                'label': example['label'],
                'correct': is_correct,
                'generated_text': generated_text,
                'inference_time': elapsed_time
            })
            
            # Log progress
            if len(results) % 10 == 0:
                current_accuracy = metrics['correct'] / metrics['total']
                logger.info(f"Strategy: {strategy}, Running accuracy: {current_accuracy:.4f}")
            
            # Log cache stats for all models using past_key_values
            # if len(results) % 10 == 0:  # Log every 10 samples
            #     stats = cache_metrics.get_stats()
            #     logger.info(f"Cache Stats: Memory={stats['total_memory_mb']:.2f}MB, "
            #                f"Avg Tokens={stats['avg_tokens_cached']:.1f}, "
            #                f"Layers={stats['num_layers']}")
            #     metrics['cache_stats'].append(stats)
    
    # Calculate final metrics
    metrics['accuracy'] = metrics['correct'] / metrics['total']
    metrics['avg_inference_time'] = sum(metrics['inference_times']) / len(metrics['inference_times'])
    metrics['avg_memory_usage'] = sum(metrics['memory_usage']) / len(metrics['memory_usage']) if metrics['memory_usage'] else None
    
    # Save strategy-specific results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"copa_results_{strategy}_{timestamp}.jsonl"
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'strategy': strategy,
        'accuracy': metrics['accuracy'],
        'avg_inference_time': metrics['avg_inference_time'],
        'avg_memory_usage': metrics['avg_memory_usage'],
        'total_samples': metrics['total'],
        'results_file': output_path
    }

def run_benchmark(args):
    """Run benchmark across all specified strategies"""
    if not hasattr(args, 'strategies'):
        logger.warning("No strategies specified in config, using only the configured attention_type")
        strategies_to_run = [args.attention_type]
    else:
        strategies_to_run = args.strategies
    
    logger.info(f"Running benchmark with strategies: {strategies_to_run}")
    
    strategies = {
        "full": lambda c: setattr(c, "attention_type", "default"),
        "streaming": lambda c: {
            setattr(c, "attention_type", "streaming"),
            setattr(c, "start_size", c.start_size),
            setattr(c, "recent_size", c.recent_size)
        },
        "h2o": lambda c: setattr(c, "attention_type", "heavy_hitter"),
        "sparse_fixed": lambda c: {
            setattr(c, "attention_type", "sparse_fixed"),
            setattr(c, "window_size", args.window_size if hasattr(args, 'window_size') else 256)
        },
        "sparse_strided": lambda c: {
            setattr(c, "attention_type", "sparse_strided"),
            setattr(c, "window_size", args.window_size if hasattr(args, 'window_size') else 256),
            setattr(c, "stride", args.stride if hasattr(args, 'stride') else 128)
        },
        "liquid_fusion": lambda c: {
            setattr(c, "attention_type", "liquid_fusion"),
            setattr(c, "start_size", args.start_size if hasattr(args, 'start_size') else 4),
            setattr(c, "recent_size", args.recent_size if hasattr(args, 'recent_size') else 64),
            setattr(c, "heavy_ratio", args.heavy_ratio if hasattr(args, 'heavy_ratio') else 0.1),
            setattr(c, "recent_ratio", args.recent_ratio if hasattr(args, 'recent_ratio') else 0.1)
        }
    }
    
    all_results = {}
    
    for strategy_name in strategies_to_run:
        logger.info(f"\nTesting {strategy_name} strategy...")
        config = copy.deepcopy(args)
        
        # Apply strategy configuration
        strategy_fn = strategies[strategy_name]
        if isinstance(strategy_fn(config), dict):
            logger.info(f"Applied {strategy_name} configuration with start_size={config.start_size}, recent_size={config.recent_size}")
        else:
            logger.info(f"Applied {strategy_name} configuration")
        
        logger.info(f"Config attention_type set to: {config.attention_type}")
        
        # Clear any existing cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()
        
        try:
            result = run_single_strategy_benchmark(
                config,
                strategy=strategy_name
            )
            all_results[strategy_name] = result
            
            # Force cleanup after each strategy
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                
        except Exception as e:
            logger.error(f"Error in {strategy_name}: {str(e)}")
            all_results[strategy_name] = {"error": str(e)}
            logger.exception("Full traceback:")
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"copa_benchmark_summary_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Device-specific cleanup
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    gc.collect()
    
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--strategies", nargs="+", 
                       default=["full", "streaming", "h2o", "sparse_fixed", "sparse_strided", "liquid_fusion"])
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--heavy_ratio", type=float, default=0.1,
                      help="Ratio of tokens to keep as heavy hitters (default: 0.1 for 10%)")
    parser.add_argument("--recent_ratio", type=float, default=0.1,
                      help="Ratio of recent tokens to keep (default: 0.1 for 10%)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    results = run_benchmark(args)
