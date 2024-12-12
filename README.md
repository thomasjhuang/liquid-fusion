# LiquidFusion

LiquidFusion is an attention mechanism that blends the ideas of H2O (Heavy Hitter Oracle for Efficient Generative Inference of Large Language Models) and 
StreamingLLM to create an efficient hybrid attention pattern for LLMs.

## Installation

```bash
git clone https://github.com/thomasjhuang/liquid-fusion.git
cd liquid-fusion
pip install -r requirements.txt
```

## Directory Structure

```bash
liquid-fusion/
├── configs/
│   ├── experiment.yaml
│   ├── model_configs.py
│   └── custom/
│       └── fast_test.yaml
├── data/
│   ├── datasets.py
│   └── preprocessing.py
├── experiments/
│   └── evaluate.py
├── models/
│   ├── attention/
│   │   ├── liquid_fusion.py
│   │   ├── h2o.py
│   │   └── streaming.py
│   └── base_models.py
├── scripts/
│   ├── run_experiment.py
│   └── run_benchmark.py
└── utils/
    └── visualization.py
```

## Benchmarking

### Running Benchmarks

The benchmark script supports multiple attention strategies and cache sizes:

```bash
python -m scripts.run_benchmark \
    --model "huggyllama/llama-7b" \
    --strategy "liquid_fusion" \
    --cache-size 80 \
    --device "cuda"
```

### Available Options

- `--model`: Model identifier (default: "huggyllama/llama-7b")
- `--strategy`: Attention strategy ["full", "streaming", "h2o", "liquid_fusion"]
- `--device`: Computing device ["cuda", "cpu", "mps"]
- `--max-tokens`: Maximum tokens to generate (default: 32)

### Attention Parameters

LiquidFusion specific parameters:
- `window_size`: Size of local attention window (default: 256)
- `sink_size`: Number of sink tokens (default: 4)
- `sink_update_rate`: Rate for updating sink tokens (default: 0.1)
- `heavy_ratio`: Ratio for heavy hitter selection (default: 0.2)
- `recent_ratio`: Ratio for recent token selection (default: 0.3)

### Example Benchmark Commands

1. Full baseline attention:
```bash
python -m scripts.run_benchmark --strategy full --cache-size 100
```

2. LiquidFusion with reduced memory:
```bash
python -m scripts.run_benchmark --strategy liquid_fusion --cache-size 40
```

3. Compare multiple strategies:
```bash
for strategy in full streaming h2o liquid_fusion local; do
    for cache in 4 20 40 80 100; do
        python -m scripts.run_benchmark \
            --strategy $strategy \
            --cache-size $cache
    done
done
```

### Benchmark Results

Results are saved in JSON format under `benchmark_results/` with the naming convention:
```
benchmark_results_{strategy}_{model}_{cache_size}_{timestamp}.json
```

Example result structure:
```json
{
  "metadata": {
    "model_name": "huggyllama/llama-7b",
    "strategy": "liquid_fusion",
    "cache_size": 80,
    "attention_type": "default",
    "window_size": 256,
    "sink_size": 4,
    "sink_update_rate": 0.1,
    "heavy_ratio": 0.2,
    "recent_ratio": 0.3
  },
  "results": [
    {
      "request": {
        "prompt": "...",
        "temperature": 0.7,
        "max_tokens": 32
      },
      "result": {
        "choices": [...],
        "request_time": {
          "batch_time": 4.15,
          "batch_size": 1
        }
      }
    }
  ]
}
```

## Usage (General)

Run experiments:
```bash
python -m liquid-fusion.scripts.run_experiment
```

With custom config:
```bash
python -m liquid-fusion.scripts.run_experiment experiment=custom/fast_test
```

## Citation

If you use LiquidFusion in your research, please cite:

```bibtex
@article{huang2024liquidfusion,
  title={LiquidFusion: A Mixed Attention Approach for Efficient LLM Inference},
  author={Huang, Thomas},
  journal={arXiv preprint},
  year={2024}
}
```
