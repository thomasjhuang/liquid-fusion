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
        # Clear memory before starting
        gc.collect()
        torch.cuda.empty_cache()
        
        # Set memory allocation strategy
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.9)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Load model and tokenizer
        log("\nInitializing components...")
        model_loader = ModelLoader(config)
        with torch.cuda.amp.autocast():
            model, tokenizer = model_loader.load_model_and_tokenizer()
            model = model.to(device)
            
            # Initialize dataset manager with tokenizer
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
                                
                                result = process_outputs(outputs, input_text, tokenizer, start_time, config, device)
                                results.append(result)
                                del outputs
                                
                except Exception as e:
                    log(f"Error testing {dataset_config.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
    finally:
        # Clean up
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
    
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

