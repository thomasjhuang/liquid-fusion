import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from data.config import BenchmarkConfig
from scripts.run_benchmark import run_benchmark


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_budget_sweep(args):
    results = {}
    baseline = BenchmarkConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        strategies=["full"],
        sequence_length=args.sequence_length,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )
    full_result = run_benchmark(baseline)["full"]
    if "error" in full_result:
        raise RuntimeError(full_result["error"])
    results["full"] = full_result

    for capacity in args.cache_capacities:
        sink_size = min(args.sink_size, capacity - 1)
        remaining = capacity - sink_size
        heavy_budget = remaining // 2
        recent_budget = remaining - heavy_budget
        config = BenchmarkConfig(
            model_name=args.model,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            strategies=["liquid_fusion"],
            sequence_length=args.sequence_length,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            sink_size=sink_size,
            heavy_budget=heavy_budget,
            recent_budget=recent_budget,
        )
        liquid_result = run_benchmark(config)["liquid_fusion"]
        if "error" in liquid_result:
            raise RuntimeError(liquid_result["error"])
        results[str(capacity)] = liquid_result

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cache_budget_sweep_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_revision")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument(
        "--cache_capacities",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024],
    )
    args = parser.parse_args()
    if any(capacity < 2 for capacity in args.cache_capacities):
        parser.error("cache capacities must be at least 2")
    return args


if __name__ == "__main__":
    print(json.dumps(run_budget_sweep(parse_args()), indent=2))
