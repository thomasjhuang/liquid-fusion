import torch
import numpy as np
import time
from typing import Dict, DefaultDict
from tqdm import tqdm
from threading import Lock
from collections import defaultdict

class BenchmarkMetrics:
    
    @staticmethod
    def run_helm_evaluation(model, tokenizer, scenarios=None):
        """
        Run HELM evaluation suite on the model
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            scenarios: List of HELM scenario names to run (None for all)
        """
        # Create HELM runner
        runner = Runner(model=model, tokenizer=tokenizer)
        
        # Get scenarios to run
        if scenarios is None:
            scenarios = [
                "mmlu:subject=humanities,subset=5shot",
                "truthfulqa:task=mc",
                "bbq:task=mc",
                "hellaswag",
                "winogrande"
            ]
            
        results = {}
        for scenario in scenarios:
            # Run scenario
            scenario_results = runner.run_scenario(scenario)
            
            # Calculate metrics
            metrics = get_metrics(scenario_results)
            
            results[scenario] = {
                "accuracy": metrics.get("accuracy"),
                "perplexity": metrics.get("perplexity"),
                "latency": metrics.get("latency_ms"),
                "throughput": metrics.get("throughput")
            }
            
        return results

    @staticmethod
    def measure_perplexity(model, dataloader, batch_size) -> float:
        model.eval()
        total_loss = 0
        total_tokens = 0
        device = next(model.parameters()).device
        
        # Create progress bar
        pbar = tqdm(total=len(dataloader), desc=f"Measuring perplexity (batch_size={batch_size})", unit="batch")

        with torch.no_grad():
            for batch in dataloader:
                # Move everything to GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else None

                model_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                if labels is not None:
                    model_inputs['labels'] = labels

                outputs = model(**model_inputs)
                loss = outputs.loss

                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # Update progress bar with current perplexity
                current_ppl = torch.exp(torch.tensor(total_loss / total_tokens))
                pbar.set_postfix({'perplexity': f'{current_ppl:.2f}'})
                pbar.update(1)

        pbar.close()
        return torch.exp(torch.tensor(total_loss / total_tokens))

    @staticmethod
    def measure_latency_throughput(model, dataloader, batch_size) -> Dict[str, float]:
        model.eval()
        latencies = []
        total_tokens = 0
        start_time = time.time()
        device = next(model.parameters()).device
        
        # Create progress bar
        pbar = tqdm(total=len(dataloader), desc=f"Measuring latency (batch_size={batch_size})", unit="batch")

        with torch.no_grad():
            for batch in dataloader:
                # Move everything to GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else None

                model_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                if labels is not None:
                    model_inputs['labels'] = labels

                batch_start = time.time()
                _ = model(**model_inputs)
                batch_latency = time.time() - batch_start
                latencies.append(batch_latency)

                total_tokens += attention_mask.sum().item()

                # Update progress bar with current latency
                avg_latency = np.mean(latencies) * 1000  # Convert to ms
                pbar.set_postfix({'avg_latency': f'{avg_latency:.2f}ms'})
                pbar.update(1)

        pbar.close()
        total_time = time.time() - start_time

        return {
            "avg_latency": np.mean(latencies),
            "p90_latency": np.percentile(latencies, 90),
            "p99_latency": np.percentile(latencies, 99),
            "throughput": total_tokens / total_time
        }

    @staticmethod
    def measure_kv_cache(model) -> Dict[str, int]:
        total_params = 0
        cache_params = 0

        # Create progress bar for model parameters
        pbar = tqdm(model.named_parameters(), desc="Measuring KV cache size", unit="param")

        for name, param in pbar:
            if "k_proj" in name or "v_proj" in name:
                cache_params += param.numel()
            total_params += param.numel()

            # Update progress bar with current cache size
            cache_size_mb = cache_params * 4 / (1024 * 1024)  # Assuming float32
            pbar.set_postfix({'cache_size': f'{cache_size_mb:.2f}MB'})

        pbar.close()

        return {
            "total_params": total_params,
            "cache_params": cache_params,
            "cache_ratio": cache_params / total_params,
            "cache_size_mb": cache_params * 4 / (1024 * 1024)  # Assuming float32
        }

class CacheMetrics:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CacheMetrics, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.reset()
            self._initialized = True
            self._lock = Lock()
    
    def reset(self):
        """Reset all metrics"""
        self.layer_metrics = defaultdict(lambda: {
            'memory_usage': 0,
            'tokens_cached': 0,
            'max_length': 0
        })
    
    def update_from_past_key_values(self, past_key_values):
        """Update metrics from model's past_key_values"""
        if past_key_values is None:
            return
            
        with self._lock:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                layer_id = f"layer_{layer_idx}"
                
                # Calculate memory usage in MB
                memory_mb = (
                    (key_states.nelement() * key_states.element_size() + 
                     value_states.nelement() * value_states.element_size()) 
                    / (1024 * 1024)
                )
                
                self.layer_metrics[layer_id].update({
                    'memory_usage': memory_mb,
                    'tokens_cached': key_states.shape[1],  # seq_len dimension
                    'max_length': max(self.layer_metrics[layer_id]['max_length'], key_states.shape[1])
                })
    
    def get_stats(self) -> Dict:
        """Get current cache statistics"""
        with self._lock:
            total_memory = sum(m['memory_usage'] for m in self.layer_metrics.values())
            avg_tokens = np.mean([m['tokens_cached'] for m in self.layer_metrics.values()]) if self.layer_metrics else 0
            
            return {
                'total_memory_mb': total_memory,
                'avg_tokens_cached': avg_tokens,
                'num_layers': len(self.layer_metrics),
                'per_layer': dict(self.layer_metrics)
            }
