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

def get_generation_config(tokenizer, input_ids, config):
    """Standardized generation configuration with proper attention mask"""
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones_like(input_ids)
    
    # Ensure tokenizer has proper padding settings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': config.max_tokens,
        'do_sample': True if config.temperature > 0 else False,
        'temperature': config.temperature,
        'num_return_sequences': 1,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
        'max_length': config.sequence_length + config.max_tokens,
        'output_attentions': False,
        'output_hidden_states': False,
        'return_dict': True,
    }

def run_single_strategy_benchmark(config, strategy):
    """Run benchmark for a single attention strategy"""
    logger.info(f"\n=== Starting benchmark for strategy: {strategy} ===")
    
    # Load model
    logger.info("Loading model and tokenizer...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    # Configure tokenizer
    logger.info("Configuring tokenizer...")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    logger.info("Loading COPA dataset...")
    dataset = load_dataset("super_glue", "copa", split="validation")
    if config.max_samples:
        dataset = dataset.select(range(config.max_samples))
    logger.info(f"Dataset size: {len(dataset)}")

    results = []
    metrics = {
        'correct': 0,
        'total': 0,
        'accuracy': 0.0,
        'memory_usage': [],
        'inference_times': [],
        'cache_stats': []
    }

    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {strategy}")):
            logger.info(f"\nProcessing example {idx}")
            
            # Format prompt
            premise = example['premise']
            choice1 = example['choice1']
            choice2 = example['choice2']
            question = "cause" if example['question'] == 0 else "effect"
            prompt = f"Given the {question}: '{premise}'\nWhich is more likely?\n1. {choice1}\n2. {choice2}\nAnswer:"
            logger.info(f"Formatted prompt: {prompt[:100]}...")
            
            # Tokenize
            logger.info("Tokenizing input...")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            logger.info(f"Input shape: {input_ids.shape}")
            
            # Generate
            logger.info("Preparing generation config...")
            generation_config = get_generation_config(tokenizer, input_ids, config)
            logger.info(f"Generation config: {generation_config}")
            
            try:
                logger.info("Starting generation...")
                output = model.generate(**generation_config)
                logger.info(f"Generation successful, output shape: {output.shape}")
            except Exception as e:
                logger.error(f"Generation failed with error: {str(e)}")
                logger.error(f"Model type at failure: {type(model)}")
                logger.error(f"Model forward method: {getattr(model, 'forward', None)}")
                raise
            
            # Process output
            generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            logger.info(f"Generated text: {generated_text}")
            
            # Rest of the processing...

    return {
        'strategy': strategy,
        'accuracy': metrics['accuracy'],
        'avg_inference_time': sum(metrics['inference_times']) / len(metrics['inference_times']),
        'avg_memory_usage': sum(metrics['memory_usage']) / len(metrics['memory_usage']) if metrics['memory_usage'] else None,
        'total_samples': metrics['total'],
        'results_file': f"copa_results_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
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
