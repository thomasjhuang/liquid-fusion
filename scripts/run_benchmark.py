import argparse
import copy
import gc
import json
import logging
import platform
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import transformers
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

from data.config import BenchmarkConfig, DatasetConfig
from models.base_models import ModelLoader


logger = logging.getLogger(__name__)


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _clear_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def _cache_capacity(config: BenchmarkConfig, strategy: str):
    if strategy == "streaming":
        return config.start_size + config.recent_size
    if strategy == "h2o":
        return config.heavy_budget + config.recent_budget
    if strategy == "liquid_fusion":
        return config.sink_size + config.heavy_budget + config.recent_budget
    return None


def _load_benchmark_dataset(dataset_config: DatasetConfig, max_samples: int):
    kwargs = {
        "path": dataset_config.name,
        "split": dataset_config.splits[0],
        "streaming": True,
    }
    if dataset_config.config:
        kwargs["name"] = dataset_config.config
    if dataset_config.revision:
        kwargs["revision"] = dataset_config.revision
    return load_dataset(**kwargs).take(
        dataset_config.max_samples or max_samples
    )


def _tokenize_prompt(
    tokenizer,
    dataset_config: DatasetConfig,
    sample,
    max_input_tokens: int,
    device: str,
):
    prefix = tokenizer.encode(
        dataset_config.input_prefix,
        add_special_tokens=True,
    )
    suffix = tokenizer.encode(
        dataset_config.output_prefix,
        add_special_tokens=False,
    )
    document = tokenizer.encode(
        str(sample[dataset_config.input_field]),
        add_special_tokens=False,
    )
    available_document_tokens = max_input_tokens - len(prefix) - len(suffix)
    if available_document_tokens < 1:
        raise ValueError("sequence_length is too small for the benchmark prompt")
    input_ids = torch.tensor(
        [prefix + document[:available_document_tokens] + suffix],
        dtype=torch.long,
        device=device,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def run_single_strategy_benchmark(
    config: BenchmarkConfig,
    strategy: str,
) -> Dict[str, float]:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model, tokenizer = ModelLoader(config).load_model_and_tokenizer()
    tokenizer.padding_side = "left"
    dataset_config = config.datasets[0]
    dataset = _load_benchmark_dataset(dataset_config, config.max_samples)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rouge_scores = []
    inference_times = []
    generated_tokens = 0
    input_lengths = []
    model_context = getattr(
        model.config,
        "max_position_embeddings",
        config.sequence_length + config.max_new_tokens,
    )
    max_input_tokens = min(
        config.sequence_length,
        model_context - config.max_new_tokens,
    )
    if max_input_tokens < 1:
        raise ValueError("max_new_tokens leaves no room for an input prompt")

    if config.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for sample in tqdm(dataset, desc=f"Evaluating {strategy}"):
        inputs = _tokenize_prompt(
            tokenizer,
            dataset_config,
            sample,
            max_input_tokens,
            config.device,
        )
        input_length = inputs["input_ids"].shape[1]

        _synchronize(config.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        _synchronize(config.device)
        elapsed = time.perf_counter() - started_at

        new_tokens = output[:, input_length:]
        prediction = tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        )
        reference = sample[dataset_config.reference_field]
        rouge_scores.append(scorer.score(reference, prediction))
        inference_times.append(elapsed)
        generated_tokens += new_tokens.shape[1]
        input_lengths.append(input_length)

    total_time = sum(inference_times)
    count = len(rouge_scores)
    peak_memory = (
        torch.cuda.max_memory_allocated()
        if config.device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    result = {
        "samples": count,
        "avg_rouge1": sum(score["rouge1"].fmeasure for score in rouge_scores) / count,
        "avg_rouge2": sum(score["rouge2"].fmeasure for score in rouge_scores) / count,
        "avg_rougeL": sum(score["rougeL"].fmeasure for score in rouge_scores) / count,
        "avg_generation_time_seconds": total_time / count,
        "output_tokens_per_second": generated_tokens / total_time,
        "avg_input_tokens": sum(input_lengths) / count,
        "min_input_tokens": min(input_lengths),
        "max_input_tokens": max(input_lengths),
        "generated_tokens": generated_tokens,
        "peak_device_memory_bytes": peak_memory,
        "configured_cache_tokens": _cache_capacity(config, strategy),
        "attention_backend": "sdpa" if strategy == "full" else "eager_reference",
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "dataset_name": dataset_config.name,
        "dataset_revision": dataset_config.revision,
        "dataset_split": dataset_config.splits[0],
        "strategy": strategy,
        "device": config.device,
        "dtype": config.dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "configuration": asdict(config),
    }

    del model
    _clear_device(config.device)
    return result


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Dict[str, float]]:
    strategy_names = config.strategies or [config.attention_type]
    supported = {"full", "streaming", "h2o", "liquid_fusion"}
    unknown = set(strategy_names) - supported
    if unknown:
        raise ValueError(f"Unsupported strategies: {sorted(unknown)}")

    results = {}
    for strategy in strategy_names:
        strategy_config = copy.deepcopy(config)
        strategy_config.attention_type = "default" if strategy == "full" else strategy
        logger.info("Running strategy %s", strategy)
        try:
            results[strategy] = run_single_strategy_benchmark(
                strategy_config,
                strategy,
            )
        except Exception as error:
            logger.exception("Strategy %s failed", strategy)
            results[strategy] = {"error": str(error)}
        _clear_device(config.device)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"benchmark_{timestamp}.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", result_path)
    return results


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--model_name", dest="model_name", required=True)
    parser.add_argument("--model_revision")
    parser.add_argument(
        "--strategies",
        "--strategy",
        nargs="+",
        default=["full", "liquid_fusion"],
        choices=["full", "streaming", "h2o", "liquid_fusion"],
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=512)
    parser.add_argument("--heavy_budget", type=int, default=256)
    parser.add_argument("--recent_budget", type=int, default=256)
    parser.add_argument("--dataset_name", default="EdinburghNLP/xsum")
    parser.add_argument("--dataset_config")
    parser.add_argument("--dataset_revision")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--input_field", default="document")
    parser.add_argument("--reference_field", default="summary")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    dataset = DatasetConfig(
        name=args.dataset_name,
        config=args.dataset_config,
        revision=args.dataset_revision,
        splits=[args.dataset_split],
        input_field=args.input_field,
        reference_field=args.reference_field,
    )
    return BenchmarkConfig(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        strategies=args.strategies,
        sequence_length=args.sequence_length,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        sink_size=args.sink_size,
        start_size=args.start_size,
        recent_size=args.recent_size,
        heavy_budget=args.heavy_budget,
        recent_budget=args.recent_budget,
        datasets=[dataset],
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark_results = run_benchmark(parse_args())
    if any("error" in result for result in benchmark_results.values()):
        raise SystemExit(1)
