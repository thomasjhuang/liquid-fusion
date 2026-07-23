# LiquidFusion

LiquidFusion is a bounded KV-cache policy for Llama-family inference. It retains attention sinks, cumulative heavy hitters, and a rolling recent window while preserving native grouped-query-attention storage and absolute RoPE positions.

This repository provides a correctness reference implementation. Dense prompt prefill is still quadratic, and the Python cache policy is not yet a production serving kernel.

## What is implemented

- A physical cache bound during prefill and decoding.
- Sink, heavy, and recent cache partitions.
- Normalized causal-attention importance scores.
- Native KV-head storage for GQA models.
- Absolute positions that do not reset after eviction.
- StreamingLLM-compatible and H2O-compatible bounded policy configurations.
- Offline correctness validation using a randomly initialized tiny Llama.
- XSum quality, throughput, and peak-memory benchmarking.

## Install

```bash
python -m pip install -e ".[dev]"
```

Transformers 4.36.0 is pinned because the custom cache API is version-sensitive.

## Validate without downloading a model

```bash
python -m scripts.validate_small_model
python -m pytest -q
```

The validation checks:

- Dense prefill logits match unmodified Llama.
- Cache length never exceeds its configured capacity.
- KV tensors retain `num_key_value_heads`, not expanded query heads.
- Absolute token positions continue increasing after eviction.

This is sufficient to validate mechanics on a CPU. It does not establish model quality.

## Validate on a small pretrained model

TinyLlama 1.1B is small enough for a single consumer GPU and can run on CPU with reduced settings:

```bash
python -m scripts.run_benchmark \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --strategies full liquid_fusion \
  --device cuda \
  --dtype float16 \
  --sequence_length 2048 \
  --max_new_tokens 64 \
  --max_samples 10 \
  --sink_size 4 \
  --heavy_budget 256 \
  --recent_budget 256
```

For CPU validation:

```bash
python -m scripts.run_benchmark \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --strategies full liquid_fusion \
  --device cpu \
  --dtype float32 \
  --sequence_length 512 \
  --max_new_tokens 16 \
  --max_samples 2 \
  --sink_size 4 \
  --heavy_budget 64 \
  --recent_budget 64
```

Results are written to `results/benchmark_<timestamp>.json`.

TinyLlama can expose correctness regressions and large quality failures. It is not sufficient for a state-of-the-art claim. Final evaluation should use multiple modern 7B–8B long-context checkpoints and matched physical cache budgets.

The benchmark reserves `max_new_tokens` inside the model context window and truncates only document tokens, preserving the summarization instruction. Its output-token throughput includes both prefill and decode. Full attention uses SDPA while compressed policies use the eager correctness backend, so these numbers validate behavior but are not a serving-speed claim.

## Strategies

| Strategy | Cache policy |
| --- | --- |
| `full` | Unmodified model cache |
| `streaming` | Sink plus recent window |
| `h2o` | Heavy plus recent tokens |
| `liquid_fusion` | Sink plus heavy plus recent tokens |

All compressed strategies use the same bounded cache implementation so comparisons do not inherit the previous unbounded H2O and incorrect StreamingLLM cache behavior.

## Current limitations

- Llama-family eager attention only.
- Unpadded batches.
- One-token autoregressive decode after prefill.
- Greedy and sampling generation only; beam and contrastive generation are not supported.
- Dense quadratic prefill.
- Python token gathering rather than fused paged kernels.
- No per-layer or per-head budget allocation yet.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the hierarchical cache, serving-kernel, learned-policy, and hybrid linear-attention plans.
