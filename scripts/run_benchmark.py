from datetime import datetime
import json
from typing import Dict
import time
import torch.nn.functional as F
import torch

# Local imports
from data.config import BenchmarkConfig
from models.base_models import ModelLoader
from data.data import DatasetManager
from data.metrics import BenchmarkMetrics

import gc

def run_benchmark(config: BenchmarkConfig, 
                 save_results: bool = True,
                 verbose: bool = True) -> dict:
    """Run benchmarks with given configuration."""
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    results = []
    
    # Initialize components with memory management
    log("\nInitializing components...")
    try:
        # Clear memory before loading model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load model with memory optimizations
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_model_and_tokenizer()
        
        # Enable memory efficient options
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # Process datasets one at a time
        for dataset_config in config.datasets:
            for split in dataset_config.splits:
                log(f"\nTesting {dataset_config.name} ({split})")
                try:
                    # Clear memory before loading each dataset
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Initialize dataset manager for just this dataset
                    dataset_manager = DatasetManager(config, tokenizer)
                    dataset = dataset_manager.load_dataset(
                        dataset_config.name, 
                        split,
                        batch_size=1
                    )
                    
                    if dataset is None:
                        log(f"Skipping {dataset_config.name} - failed to load")
                        continue
                    
                    # Process each example
                    for batch in dataset:
                        start_time = time.time()
                        
                        # Move batch to GPU efficiently
                        input_text = batch['text']
                        input_ids = batch['input_ids'].to(model.device, non_blocking=True)
                        attention_mask = batch['attention_mask'].to(model.device, non_blocking=True)
                        
                        # Generate with memory optimization
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            current_length = input_ids.shape[1]
                            max_new_tokens = min(
                                config.max_tokens,
                                model.config.max_position_embeddings - current_length
                            )
                            
                            outputs = model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=config.temperature,
                                return_dict_in_generate=True,
                                output_scores=True,
                                pad_token_id=tokenizer.pad_token_id,
                                return_legacy_cache=True
                            )
                        
                        # Process results and clear memory
                        result = process_outputs(outputs, input_text, tokenizer, start_time, config)
                        results.append(result)
                        
                        # Clear memory after each generation
                        del outputs
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    log(f"Error testing {dataset_config.name}: {str(e)}")
                    continue
                
                # Clear dataset from memory
                del dataset
                gc.collect()
                torch.cuda.empty_cache()
        
    finally:
        # Clean up
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results if requested
    if save_results and results:
        save_results_to_file(results, config)
    
    return results

def process_outputs(outputs, input_text, tokenizer, start_time, config):
    """Process model outputs into result format"""
    generated_tokens = outputs.sequences[0, -config.max_tokens:]
    tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
    
    # Calculate log probabilities
    logprobs = []
    top_logprobs = []
    for logits in outputs.scores:
        probs = F.softmax(logits[0], dim=-1)
        log_probs = torch.log(probs)
        top_prob, top_idx = probs.max(dim=-1)
        logprobs.append(log_probs[top_idx].item())
        top_token = tokenizer.convert_ids_to_tokens(top_idx.item())
        top_logprobs.append({top_token: log_probs[top_idx].item()})
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
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

