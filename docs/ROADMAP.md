# LiquidFusion roadmap

## Product boundary

LiquidFusion has two independent tracks:

1. A training-free, bounded KV-cache policy for existing Llama checkpoints.
2. A distilled hybrid recurrent/softmax model for experiments with linear attention.

The first track must be validated before work begins on the second. A linear-attention operator cannot safely replace softmax attention in an unchanged checkpoint.

## Implemented foundation

- Native GQA KV storage without expanding KV heads.
- Absolute RoPE positions independent of physical cache length.
- A hard cache capacity at every layer and decode step.
- Sink, cumulative-heavy, and recent token partitions.
- Heavy scores derived from normalized causal attention.
- Physically bounded StreamingLLM and H2O policy configurations.
- Dense prefill equivalence with the original model.
- A reproducible CPU validation model with no checkpoint download.
- A benchmark CLI, cache metrics, tests, and CI.

Dense prefill remains quadratic. The current implementation is a correctness reference rather than the final serving kernel.

## Stage 1: correctness and benchmark gates

- Validate Llama and TinyLlama checkpoints at full-cache and compressed budgets.
- Add tests for batch sizes greater than one without padding.
- Add explicit handling for padded batches.
- Add a Transformers-version compatibility matrix.
- Record quality, retained KV bytes, prefill latency, decode latency, and peak memory separately.
- Run LongBench, RULER, SCBench, retrieval, and summarization tasks.
- Compare at matched physical cache bytes against full attention, StreamingLLM, H2O, SnapKV, PyramidKV, Ada-KV, RazorAttention, and quantized KV.

Exit gate: reproducible quality-versus-memory curves and no correctness invariant failures.

## Stage 2: hierarchical policy

- Allocate cache budgets per layer rather than uniformly.
- Allocate budgets per KV head or compatible head group.
- Identify retrieval heads and retain wider history only for those heads.
- Add compensation summaries for evicted local-head history.
- Score semantic chunks first and tokens second.
- Refresh historical candidates periodically rather than on every token.
- Add question-aware and question-agnostic policy modes.

The controller should optimize under a byte and latency budget, not a token-count budget alone.

Exit gate: statistically significant quality improvements over the best matched-budget baseline on both retrieval and reasoning tasks.

## Stage 3: serving backend

- Move KV storage into fixed-size pages.
- Keep recent tokens in a ring buffer.
- Use FlashAttention or SDPA for dense and local attention.
- Add fused indexed or block-sparse decode kernels using FlashInfer or Triton.
- Quantize historical pages independently from recent pages.
- Support continuous batching and beam cache reordering.
- Port the policy interface to vLLM or SGLang.

Exit gate: end-to-end throughput and latency improve after including policy-selection and gather overhead.

## Stage 4: learned policy

- Distill token-retention decisions from full attention.
- Train a small layer/head budget controller.
- Compare learned selection with deterministic Ada-KV-style allocation.
- Retain a deterministic fallback and measure controller overhead.

Exit gate: learned selection improves the quality-versus-latency frontier across unseen tasks and context lengths.

## Stage 5: hybrid linear-attention track

- Start from a separate pretrained teacher checkpoint.
- Replace selected attention blocks with GatedDeltaNet or HGRN-2-style recurrent blocks.
- Keep periodic global softmax layers and local softmax windows.
- Use blockwise output matching before sequence-level distillation.
- Search the placement of retained softmax layers instead of choosing a uniform pattern.
- Train and evaluate positional handling for lengths beyond the teacher context.
- Compare against the KV-compressed teacher at equal quality, memory, and throughput.

This track requires training data and accelerator capacity. It must not be represented as a drop-in cache optimization.

## Required measurements

Every experiment must report:

- Model and exact revision.
- Dataset and exact revision.
- Prompt and output length distributions.
- Cache bytes by layer and head.
- Realized compression ratio.
- Time to first token.
- Inter-token latency.
- Decode throughput.
- Peak device memory.
- Policy update and gather overhead.
- Task quality with confidence intervals.
- Hardware, software versions, and random seeds.

Compression ratio alone is not evidence of a serving improvement.
