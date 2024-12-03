import torch
import numpy as np
import time
from typing import Dict
from tqdm import tqdm

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
