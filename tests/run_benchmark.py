from datetime import datetime
import json
from typing import Dict

# Local imports
from data.config import BenchmarkConfig
from models.base_models import ModelLoader
from data.data import DatasetManager
from data.metrics import BenchmarkMetrics

# HELM imports
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.runner import RunSpec, Runner
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.common.general import ensure_directory_exists
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.executor import ExecutionSpec
from helm.common.hierarchical_logger import hlog
from helm.common.authentication import Authentication
from helm.common.cache_backend_config import SqliteCacheBackendConfig
from helm.benchmark.model_deployment_registry import (
    ModelDeployment,
    register_model_deployment,
)
from helm.common.object_spec import ObjectSpec
from helm.benchmark.huggingface_registration import register_huggingface_model

class ClientSpec(ObjectSpec):
    pass

class WindowServiceSpec(ObjectSpec):
    pass

def register_local_model(model_name: str, model_type: str, **kwargs):
    """
    Register a local model with HELM's deployment registry.
    
    Args:
        model_name: Name of the model (e.g., "meta-llama/Llama-2-7b")
        model_type: Type of model (e.g., "llama", "opt", "gpt-neox")
        **kwargs: Additional arguments including:
            - max_sequence_length: Maximum sequence length (default: 2048)
            - batch_size: Batch size for inference (default: 1)
            - tokenizer_name: Name of tokenizer (default: same as model_name)
            - client_type: Type of client (default: "huggingface")
            - service_type: Type of service (default: "local")
    """
    # Default settings
    defaults = {
        'max_sequence_length': 2048,
        'batch_size': 1,
        'tokenizer_name': model_name,
        'client_type': "huggingface",
        'service_type': "local"
    }
    
    # Update defaults with any provided kwargs
    settings = {**defaults, **kwargs}
    
    # Client specification based on model type
    client_specs = {
        "llama": {
            "type": settings['client_type'],
            "engine": model_name,
            "batch_size": settings['batch_size']
        },
        "opt": {
            "type": settings['client_type'],
            "engine": model_name,
            "batch_size": settings['batch_size']
        },
        "gpt-neox": {
            "type": settings['client_type'],
            "engine": model_name,
            "batch_size": settings['batch_size']
        }
    }
    
    # Window service specifications based on model type
    service_specs = {
        "llama": {
            "type": settings['service_type'],
            "engine": model_name,
            "max_sequence_length": settings['max_sequence_length'],
            "tokenizer_name": settings['tokenizer_name']
        },
        "opt": {
            "type": settings['service_type'],
            "engine": model_name,
            "max_sequence_length": settings['max_sequence_length'],
            "tokenizer_name": settings['tokenizer_name']
        },
        "gpt-neox": {
            "type": settings['service_type'],
            "engine": model_name,
            "max_sequence_length": settings['max_sequence_length'],
            "tokenizer_name": settings['tokenizer_name']
        }
    }
    
    # Get specs for the specific model type
    client_spec = ClientSpec(**client_specs.get(model_type, client_specs["llama"]))
    window_service_spec = WindowServiceSpec(**service_specs.get(model_type, service_specs["llama"]))
    
    # Create and register the deployment
    deployment = ModelDeployment(
        name=model_name,
        client_spec=client_spec,
        model_name=model_name,
        tokenizer_name=settings['tokenizer_name'],
        window_service_spec=window_service_spec,
        max_sequence_length=settings['max_sequence_length']
    )
    register_model_deployment(deployment)

def run_helm_benchmark(config: BenchmarkConfig, save_results: bool = True, verbose: bool = True) -> dict:
    """Run custom HELM benchmarks with given configuration."""
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # register_local_model(
    #     model_name=config.model_name,
    #     model_type=config.model_type,
    #     **config.get_model_registration_args()
    # )

    register_huggingface_model(
        helm_model_name=config.model_name,
        pretrained_model_name_or_path=config.model_name
    )
    
    # Initialize execution spec with correct parameters
    execution_spec = ExecutionSpec(
        url=None,  # No proxy server needed for local execution
        auth=Authentication(api_key=""),  # Empty API key for local execution
        local_path="./helm_cache",  # Local cache directory
        parallelism=1,  # Single thread for now
        dry_run=False,
        sqlite_cache_backend_config=SqliteCacheBackendConfig(
            path="./helm_cache/sqlite"
        ),
        mongo_cache_backend_config=None  # Not using MongoDB
    )
    
    # Setup output paths
    output_path = "./helm_outputs"
    ensure_directory_exists(output_path)
    ensure_directory_exists("./helm_cache/sqlite")
    
    # Initialize HELM runner
    runner = Runner(
        execution_spec=execution_spec,
        output_path=output_path,
        suite="liquid-fusion",
        skip_instances=False,
        cache_instances=True,
        cache_instances_only=False,
        skip_completed_runs=False,
        exit_on_error=True
    )
    
    results = {
        "config": {
            "model_name": config.model_name,
            "model_type": config.model_type,
            "attention_type": config.attention_type,
        },
        "scenarios": {}
    }
    
    run_specs = []
    for dataset_config in config.datasets:
        adapter_spec_dict = dataset_config.get_adapter_spec(
            model_name=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            num_fewshot=config.num_fewshot
        )
        run_spec = RunSpec(
            name=f"{dataset_config.name}_{dataset_config.splits[0]}",
            scenario_spec=ScenarioSpec(
                class_name=dataset_config.helm_scenario,
                args=dataset_config.helm_args or {}
            ),
            adapter_spec=AdapterSpec(**adapter_spec_dict),
            metric_specs=[
                MetricSpec(
                    class_name=metric,
                    args={}
                ) for metric in config.helm_metrics
            ]
        )
        run_specs.append(run_spec)
    
    try:
        # Run all scenarios
        runner.run_all(run_specs)
        
        # Process results
        for dataset_config in config.datasets:
            scenario_path = f"{output_path}/runs/liquid-fusion/{dataset_config.name}_{dataset_config.splits[0]}"
            
            try:
                # Read stats from the output files
                with open(f"{scenario_path}/stats.json", "r") as f:
                    stats = json.load(f)
                
                results["scenarios"][dataset_config.name] = {
                    stat["name"]: stat["value"] for stat in stats
                }
                
                log(f"Completed {dataset_config.name}")
                
            except Exception as e:
                log(f"Error processing results for {dataset_config.name}: {str(e)}")
                results["scenarios"][dataset_config.name] = {"error": str(e)}
                
    except Exception as e:
        log(f"Error running benchmarks: {str(e)}")
        raise
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"helm_results_{config.model_name.split('/')[-1]}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\nResults saved to {filename}")
    
    return results

def run_benchmark(config: BenchmarkConfig, 
                 save_results: bool = True,
                 verbose: bool = True) -> dict:
    """
    Run benchmarks with given configuration.
    
    Args:
        config: BenchmarkConfig object with test settings
        save_results: Whether to save results to JSON file
        verbose: Whether to print progress and results
    
    Returns:
        Dictionary containing benchmark results
    """
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    # Initialize components
    log("\nInitializing components...")
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    dataset_manager = DatasetManager(config, tokenizer)
    metrics = BenchmarkMetrics()
    
    results = {
        "config": {
            "model_name": config.model_name,
            "model_type": config.model_type,
            "attention_type": config.attention_type,
            "batch_sizes": {
                "eval": config.eval_batch_size,
                "inference": config.inference_batch_size
            },
            "sequence_length": config.sequence_length,
            "device": config.device,
            "dtype": config.dtype
        },
        "datasets": {}
    }

    cache_budget_ratios = config.cache_budget_ratios if hasattr(config, 'cache_budget_ratios') else [0.2, 0.4, 0.6]
    
    # Run benchmarks for each dataset
    for dataset_config in config.datasets:
        for split in dataset_config.splits:
            for cache_budget_ratio in cache_budget_ratios:
                dataset_key = f"{dataset_config.name}_{split}_cache_{int(cache_budget_ratio*100)}"
                log(f"\nTesting {dataset_config.name} ({split}) with cache budget {cache_budget_ratio*100}%")
                try:
                    # Load dataset
                    dataset = dataset_manager.load_dataset(
                        dataset_config.name, 
                        split, 
                        batch_size=config.eval_batch_size
                    )
                    
                    if dataset is None:
                        log(f"Skipping {dataset_config.name} - failed to load")
                        continue
                    
                    # Apply cache budget ratio to model
                    if hasattr(model, 'set_cache_budget_ratio'):
                        model.set_cache_budget_ratio(cache_budget_ratio)
                    
                    dataset_results = {}
                    
                    # Measure metrics
                    if config.measure_perplexity:
                        perplexity = metrics.measure_perplexity(
                            model, 
                            dataset,
                            batch_size=config.eval_batch_size
                        )
                        dataset_results["perplexity"] = float(perplexity)
                        log(f"Perplexity: {perplexity:.2f}")
                    
                    
                    if config.measure_latency or config.measure_throughput:
                        latency_metrics = metrics.measure_latency_throughput(
                            model, 
                            dataset,
                            batch_size=config.inference_batch_size
                        )
                        dataset_results.update({
                            "avg_latency_ms": latency_metrics['avg_latency'] * 1000,
                            "p90_latency_ms": latency_metrics['p90_latency'] * 1000,
                            "p99_latency_ms": latency_metrics['p99_latency'] * 1000,
                            "throughput_tokens_sec": latency_metrics['throughput']
                        })
                        log(f"Average latency: {latency_metrics['avg_latency']*1000:.2f}ms")
                        log(f"P90 latency: {latency_metrics['p90_latency']*1000:.2f}ms")
                        log(f"Throughput: {latency_metrics['throughput']:.2f} tokens/sec")
                    
                    if config.measure_kv_cache:
                        cache_metrics = metrics.measure_kv_cache(model)
                        dataset_results.update({
                            "cache_size_mb": cache_metrics['cache_size_mb'],
                            "cache_ratio": cache_metrics['cache_ratio']
                        })
                        log(f"KV Cache size: {cache_metrics['cache_size_mb']:.2f}MB")
                    
                    # Measure peak memory usage
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
                    dataset_results["peak_memory_gb"] = peak_memory
                    log(f"Peak memory usage: {peak_memory:.2f}GB")
                    
                    results["datasets"][dataset_key] = dataset_results
                except Exception as e:
                    log(f"Error testing {dataset_config.name}: {str(e)}")
                    results["datasets"][dataset_key] = {"error": str(e)}
                    continue
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{config.model_name.split('/')[-1]}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\nResults saved to {filename}")
    
    return results
