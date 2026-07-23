import argparse
import copy
import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from models.attention.liquid_fusion import convert_to_liquid_fusion
from models.cache.liquid_cache import LiquidFusionCache


def build_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation_internal = "eager"
    return LlamaForCausalLM(config).eval()


def validate(
    prompt_tokens: int,
    decode_tokens: int,
    sink_size: int,
    heavy_budget: int,
    recent_budget: int,
):
    torch.manual_seed(42)
    baseline = build_model()
    liquid = copy.deepcopy(baseline)
    convert_to_liquid_fusion(
        liquid,
        sink_size=sink_size,
        heavy_budget=heavy_budget,
        recent_budget=recent_budget,
    )
    input_ids = torch.randint(1, 257, (1, prompt_tokens))

    with torch.inference_mode():
        baseline_logits = baseline(input_ids, use_cache=False).logits
        output = liquid(input_ids, use_cache=True)
        prefill_max_error = (
            output.logits - baseline_logits
        ).abs().max().item()
        cache = output.past_key_values
        observed_lengths = [cache.get_seq_length(layer) for layer in range(3)]

        for _ in range(decode_tokens):
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            output = liquid(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            observed_lengths.extend(
                cache.get_seq_length(layer) for layer in range(3)
            )

    capacity = sink_size + heavy_budget + recent_budget
    if not isinstance(cache, LiquidFusionCache):
        raise RuntimeError("Model did not return LiquidFusionCache")
    if max(observed_lengths) > capacity:
        raise RuntimeError("Cache exceeded configured capacity")
    if prefill_max_error > 1e-5:
        raise RuntimeError("LiquidFusion changed prefill logits")

    return {
        "status": "ok",
        "prefill_max_absolute_error": prefill_max_error,
        "cache_capacity_tokens": capacity,
        "max_observed_cache_tokens": max(observed_lengths),
        "total_seen_tokens": cache.seen_tokens,
        "kv_heads": cache.key_cache[0].shape[1],
        "query_heads": liquid.config.num_attention_heads,
        "last_retained_positions": cache.position_cache[0][0].tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_tokens", type=int, default=24)
    parser.add_argument("--decode_tokens", type=int, default=8)
    parser.add_argument("--sink_size", type=int, default=2)
    parser.add_argument("--heavy_budget", type=int, default=4)
    parser.add_argument("--recent_budget", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate(
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        sink_size=args.sink_size,
        heavy_budget=args.heavy_budget,
        recent_budget=args.recent_budget,
    )
    print(json.dumps(result, indent=2))
