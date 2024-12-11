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
from data.config import BenchmarkConfig
import time, copy
from datasets import load_dataset
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

def get_generation_config(tokenizer, input_ids, config):
    """Standardized generation configuration with proper attention mask"""
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones_like(input_ids)
    
    # Create GenerationConfig object instead of dictionary
    return GenerationConfig(
        max_new_tokens=config.max_tokens,
        do_sample=False,  # For deterministic outputs in benchmarking
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=True,
        output_scores=True
    ), attention_mask  # Return both config and attention mask

def calculate_accuracy(outputs, labels, tokenizer, task_type="multiple_choice"):
    """Calculate accuracy based on task type"""
    correct = 0
    total = len(outputs)
    
    if task_type == "copa":
        for output, label in zip(outputs, labels):
            # Get only the newly generated tokens
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract predicted choice (1 or 2)
            predicted_choice = None
            if '1' in response:
                predicted_choice = 0  # 0-based index
            elif '2' in response:
                predicted_choice = 1
                
            # Check if prediction matches label
            if predicted_choice is not None and predicted_choice == label:
                correct += 1

    return correct / total if total > 0 else 0.0

def run_single_strategy_benchmark(config, strategy):
    """Run benchmark for a single attention strategy"""
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"\n=== Starting benchmark for strategy: {strategy} ===")
    
    # Load model and tokenizer
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model_and_tokenizer()

    # Configure tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load Pubmed dataset instead of COPA
    dataset = load_dataset(
        config.datasets[0].name,
        "english",
        split=config.datasets[0].splits[0],
        streaming=True  # Enable streaming
    )
    # Take only max_samples
    dataset = dataset.take(config.max_samples)
    
    metrics = {
        'rouge_scores': [],
        'inference_times': [],
        'total_tokens': 0,
        'total_time': 0
    }

    outputs_list = []
    references_list = []
    
    try:
        for sample in tqdm(dataset, desc=f"Evaluating {strategy}"):
            # Format prompt for summarization
            prompt = f"Summarize the following article:\n\n{sample['text']}\n\nSummary:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
            
            # Generate summary
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Adjust based on desired summary length
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id
                )
            inference_time = time.time() - start_time
            
            # Store generated summary and reference
            new_tokens = output[:, inputs.input_ids.shape[1]:]
            outputs_list.append(tokenizer.decode(new_tokens[0], skip_special_tokens=True))
            references_list.append(sample['summary'])
            
            metrics['inference_times'].append(inference_time)
            metrics['total_tokens'] += len(new_tokens[0])
            metrics['total_time'] += inference_time

    except Exception as e:
        logger.error(f"Error in {strategy}: {str(e)}")
        raise e

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for output, reference in zip(outputs_list, references_list):
        scores = scorer.score(output, reference)
        metrics['rouge_scores'].append(scores)

    # Calculate average metrics
    avg_metrics = {
        'avg_rouge1': sum(s['rouge1'].fmeasure for s in metrics['rouge_scores']) / len(metrics['rouge_scores']),
        'avg_rouge2': sum(s['rouge2'].fmeasure for s in metrics['rouge_scores']) / len(metrics['rouge_scores']),
        'avg_rougeL': sum(s['rougeL'].fmeasure for s in metrics['rouge_scores']) / len(metrics['rouge_scores']),
        'avg_tokens_per_second': metrics['total_tokens'] / metrics['total_time'],
        'avg_inference_time': sum(metrics['inference_times']) / len(metrics['inference_times'])
    }

    return avg_metrics

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
        "h2o": lambda c: setattr(c, "attention_type", "h2o"),
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
    # Get default device
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

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
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    # Updated device argument to include MPS
    parser.add_argument("--device", type=str, 
                       default=default_device,
                       help="Device to run on (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    results = run_benchmark(args)
