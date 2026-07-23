import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from models.attention.liquid_fusion import convert_to_liquid_fusion
from models.cache.liquid_cache import LiquidFusionCache
from models.cache.policy import CacheBudget, SinkHeavyRecentPolicy


def tiny_model():
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation_internal = "eager"
    return LlamaForCausalLM(config).eval()


def test_policy_selects_sink_heavy_and_recent_tokens():
    policy = SinkHeavyRecentPolicy(CacheBudget(sink=2, heavy=2, recent=2))
    scores = torch.tensor([[0.0, 0.0, 1.0, 9.0, 8.0, 7.0, 0.0, 0.0]])
    indices = policy.select(scores)
    assert indices.tolist() == [[0, 1, 3, 4, 6, 7]]


def test_prefill_matches_full_attention_and_preserves_gqa():
    torch.manual_seed(7)
    baseline = tiny_model()
    liquid = copy.deepcopy(baseline)
    convert_to_liquid_fusion(
        liquid,
        sink_size=2,
        heavy_budget=2,
        recent_budget=3,
    )
    input_ids = torch.randint(1, 97, (1, 11))

    with torch.inference_mode():
        expected = baseline(input_ids, use_cache=False).logits
        output = liquid(input_ids, use_cache=True)

    torch.testing.assert_close(output.logits, expected, rtol=1e-5, atol=1e-5)
    cache = output.past_key_values
    assert isinstance(cache, LiquidFusionCache)
    assert cache.seen_tokens == 11
    for layer_idx in range(2):
        assert cache.key_cache[layer_idx].shape == (1, 2, 7, 8)
        positions = cache.position_cache[layer_idx][0].tolist()
        assert positions[:2] == [0, 1]
        assert positions[-3:] == [8, 9, 10]


def test_decode_cache_is_bounded_and_positions_are_absolute():
    torch.manual_seed(11)
    model = tiny_model()
    convert_to_liquid_fusion(
        model,
        sink_size=2,
        heavy_budget=2,
        recent_budget=3,
    )
    input_ids = torch.randint(1, 97, (1, 12))

    with torch.inference_mode():
        output = model(input_ids, use_cache=True)
        cache = output.past_key_values
        for _ in range(6):
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            output = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values

    assert cache.seen_tokens == 18
    for layer_idx in range(2):
        assert cache.get_seq_length(layer_idx) == 7
        positions = cache.position_cache[layer_idx][0]
        assert torch.all(positions[1:] > positions[:-1])
        assert positions[-1].item() == 17
        assert positions[-3:].tolist() == [15, 16, 17]


def test_generation_matches_when_cache_does_not_evict():
    torch.manual_seed(19)
    baseline = tiny_model()
    liquid = copy.deepcopy(baseline)
    convert_to_liquid_fusion(
        liquid,
        sink_size=2,
        heavy_budget=10,
        recent_budget=10,
    )
    input_ids = torch.randint(1, 97, (1, 8))

    with torch.inference_mode():
        expected = baseline.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
        actual = liquid.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )

    assert torch.equal(actual, expected)


def test_cache_reorders_all_metadata():
    cache = LiquidFusionCache(sink_size=1, heavy_budget=1, recent_budget=1)
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    values = keys + 100
    positions = torch.tensor([[0, 1, 2], [10, 11, 12]])
    cache.update(keys, values, 0, {"position_ids": positions})
    cache.record_attention(
        0,
        torch.tensor(
            [
                [[[0.2, 0.3, 0.5]]],
                [[[0.5, 0.3, 0.2]]],
            ]
        ),
    )

    cache.reorder_cache(torch.tensor([1, 0]))

    assert cache.position_cache[0].tolist() == [[10, 11, 12], [0, 1, 2]]
    torch.testing.assert_close(
        cache.score_cache[0],
        torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]),
    )


def test_decode_evicts_after_scoring_current_query():
    cache = LiquidFusionCache(sink_size=0, heavy_budget=1, recent_budget=2)
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    cache.update(
        keys,
        keys,
        0,
        {"position_ids": torch.tensor([[0, 1, 2]])},
    )
    cache.record_attention(
        0,
        torch.tensor([[[[0.4, 0.3, 0.3]]]]),
    )
    cache.update(
        torch.ones(1, 1, 1, 4),
        torch.ones(1, 1, 1, 4),
        0,
        {"position_ids": torch.tensor([[3]])},
    )
    assert cache.get_seq_length() == 4

    cache.record_attention(
        0,
        torch.tensor([[[[0.0, 0.8, 0.1, 0.1]]]]),
    )

    assert cache.position_cache[0].tolist() == [[1, 2, 3]]


def test_rejected_chunked_decode_does_not_advance_positions():
    cache = LiquidFusionCache(sink_size=1, heavy_budget=1, recent_budget=1)
    keys = torch.ones(1, 1, 3, 4)
    cache.update(
        keys,
        keys,
        0,
        {"position_ids": torch.tensor([[0, 1, 2]])},
    )

    try:
        cache.update(
            torch.ones(1, 1, 2, 4),
            torch.ones(1, 1, 2, 4),
            0,
            {"position_ids": torch.tensor([[3, 4]])},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Chunked decoding should fail")

    assert cache.seen_tokens == 3
    assert cache.position_cache[0].tolist() == [[0, 1, 2]]
