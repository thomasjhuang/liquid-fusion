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

def run_benchmark(config: BenchmarkConfig, 
                 save_results: bool = True,
                 verbose: bool = True) -> dict:
    """Run benchmarks with given configuration."""
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    # Initialize components
    log("\nInitializing components...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    dataset_manager = DatasetManager(config, tokenizer)
    
    results = []
    
    # Run benchmarks for each dataset
    for dataset_config in config.datasets:
        for split in dataset_config.splits:
            log(f"\nTesting {dataset_config.name} ({split})")
            try:
                # Load dataset
                dataset = dataset_manager.load_dataset(
                    dataset_config.name, 
                    split,
                    batch_size=1  # Force batch size of 1 for consistent output format
                )
                
                if dataset is None:
                    log(f"Skipping {dataset_config.name} - failed to load")
                    continue
                
                # Process each example in dataset
                for batch in dataset:
                    start_time = time.time()
                    
                    # Prepare input
                    input_text = batch['text'] if isinstance(batch['text'], str) else batch['text'][0]
                    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_length=input_ids.shape[1] + config.max_tokens,
                            do_sample=True,
                            temperature=config.temperature,
                            return_dict_in_generate=True,
                            output_scores=True,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    # Process generated tokens
                    generated_tokens = outputs.sequences[0, input_ids.shape[1]:]
                    tokens = tokenizer.convert_ids_to_tokens(generated_tokens)
                    
                    # Calculate log probabilities
                    logprobs = []
                    top_logprobs = []
                    
                    for logits in outputs.scores:
                        probs = F.softmax(logits[0], dim=-1)
                        log_probs = torch.log(probs)
                        
                        # Get top token probability
                        top_prob, top_idx = probs.max(dim=-1)
                        logprobs.append(log_probs[top_idx].item())
                        
                        # Get top token and its probability
                        top_token = tokenizer.convert_ids_to_tokens(top_idx.item())
                        top_logprobs.append({top_token: log_probs[top_idx].item()})
                    
                    # Decode generated text
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Calculate generation time
                    generation_time = time.time() - start_time
                    
                    # Format result in HELM style
                    result = {
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
                    results.append(result)
                    
            except Exception as e:
                log(f"Error testing {dataset_config.name}: {str(e)}")
                continue
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{config.model_name.split('/')[-1]}_{timestamp}.json"
        with open(filename, "w") as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        log(f"\nResults saved to {filename}")
    
    return results
