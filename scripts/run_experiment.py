import torch
import hydra
import json
from omegaconf import DictConfig
import logging
from pathlib import Path
import copy
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from ..data.datasets import AttentionDataset
from ..models.attention import LiquidFusion, H2OAttention, StreamingAttention
from ..experiments.evaluate import AttentionEvaluator
from ..configs.model_configs import LiquidFusionConfig, H2OConfig, StreamingConfig
from ..utils.visualization import plot_attention_comparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="experiment")
def run_experiment(cfg: DictConfig) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/run_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    attention_configs = {
        'liquidfusion': LiquidFusionConfig(
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            attention_sink_size=cfg.liquidfusion.attention_sink_size,
            heavy_hitter_ratio=cfg.liquidfusion.heavy_hitter_ratio
        ),
        'h2o': H2OConfig(
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            cache_ratio=cfg.h2o.cache_ratio
        ),
        'streaming': StreamingConfig(
            num_heads=cfg.model.num_heads,
            head_dim=cfg.model.head_dim,
            sink_size=cfg.streaming.sink_size
        )
    }

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    
    dataset = AttentionDataset(
        dataset_name=cfg.dataset.name,
        split=cfg.dataset.split,
        max_samples=cfg.evaluation.num_samples
    )
    
    processed_dataset = dataset.prepare_for_attention(
        tokenizer=tokenizer,
        max_length=cfg.model.max_length
    )
    
    results = {}
    attention_types = ['baseline'] + list(attention_configs.keys())
    
    for attention_type in attention_types:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )

        if attention_type != 'baseline':
            config = attention_configs[attention_type]
            attention_class = {
                'liquidfusion': LiquidFusion,
                'h2o': H2OAttention,
                'streaming': StreamingAttention
            }[attention_type]
            
            attention_module = attention_class(config).to(dtype=torch.float16)
            
            for layer in model.model.layers:
                layer.self_attn = copy.deepcopy(attention_module)

        evaluator = AttentionEvaluator(
            model=model,
            tokenizer=tokenizer,
            save_dir=experiment_dir / attention_type,
            device=cfg.hardware.device
        )

        try:
            results[attention_type] = evaluator.evaluate_model(
                dataset=processed_dataset,
                attention_type=attention_type,
                max_length=cfg.model.max_length,
                save_attention_patterns=cfg.evaluation.save_attention_patterns
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {attention_type}: {str(e)}")
            results[attention_type] = None
            
        finally:
            del model
            torch.cuda.empty_cache()
    
    try:
        plot_attention_comparison(
            results,
            save_dir=experiment_dir / "visualizations",
            save_format=cfg.visualization.format
        )
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")

    torch.save(results, experiment_dir / "final_results.pt")
    return results

def run_comprehensive_evaluation(
    models, 
    dataset, 
    num_samples=1000, 
    sequence_lengths=[128, 256, 512, 1024, 2048], 
    batch_sizes=[1, 4, 8], 
    metrics_to_track=[
        'perplexity',
        'memory_usage',
        'inference_time',
        'attention_sparsity',
        'throughput',  
        'memory_scaling', 
        'attention_patterns'  
    ]
):
    results = {model_name: {} for model_name in models.keys()}
    
    for seq_len in tqdm(sequence_lengths, desc="Sequence Lengths"):
        for batch_size in tqdm(batch_sizes, desc="Batch Sizes", leave=False):
            
            # Prepare batched data
            batched_data = []
            for i in range(0, num_samples, batch_size):
                batch = dataset.select(range(i, min(i + batch_size, num_samples)))
                batched_data.append(batch)
            
            for model_name, model in models.items():
                logger.info(f"\nEvaluating {model_name} with seq_len={seq_len}, batch_size={batch_size}")
                
                batch_metrics = []
                attention_patterns = []
                
                # Warmup pass
                warmup_text = "Warmup text" * (seq_len // 2)
                process_batch(warmup_text, model, max_length=seq_len, batch_size=batch_size)
                
                start_time = time.time()
                peak_memory = 0
                
                for batch in tqdm(batched_data, desc=f"{model_name} Processing", leave=False):
                    torch.cuda.empty_cache()
                    
                    metrics, attn_weights = process_batch(
                        batch['text'],
                        model,
                        max_length=seq_len,
                        batch_size=batch_size
                    )
                    
                    if metrics:
                        batch_metrics.append(metrics)
                        peak_memory = max(peak_memory, metrics['memory_usage'])
                        if attn_weights is not None:
                            attention_patterns.append(attn_weights)
                
                # Calculate aggregate metrics
                total_time = time.time() - start_time
                total_tokens = num_samples * seq_len
                
                aggregated_metrics = {
                    'perplexity': np.median([m['perplexity'] for m in batch_metrics]),
                    'memory_peak': peak_memory,
                    'avg_inference_time': np.mean([m['inference_time'] for m in batch_metrics]),
                    'attention_sparsity': np.mean([m['attention_sparsity'] for m in batch_metrics]),
                    'throughput': total_tokens / total_time,
                    'memory_per_token': peak_memory / (batch_size * seq_len),
                    'attention_patterns': attention_patterns
                }
                
                # Store results
                results[model_name][f'seq_{seq_len}_batch_{batch_size}'] = aggregated_metrics
                save_results(results, f'attention_results_{time.strftime("%Y%m%d_%H%M%S")}.json')
    
    return results

def save_results(results, filename):
    """Save results to a JSON file, excluding non-serializable data."""
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {}
        for config, metrics in model_results.items():
            serializable_results[model_name][config] = {
                k: v for k, v in metrics.items() 
                if k != 'attention_patterns'
            }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)

if __name__ == "__main__":
    results = run_experiment()
