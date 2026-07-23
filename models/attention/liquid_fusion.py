import math
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from models.cache.liquid_cache import LiquidFusionCache


def liquid_fusion_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[LiquidFusionCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[LiquidFusionCache]]:
    batch_size, query_length, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0,
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        query_states = torch.cat(
            [F.linear(hidden_states, query_slice) for query_slice in query_slices],
            dim=-1,
        )
        key_states = torch.cat(
            [F.linear(hidden_states, key_slice) for key_slice in key_slices],
            dim=-1,
        )
        value_states = torch.cat(
            [F.linear(hidden_states, value_slice) for value_slice in value_slices],
            dim=-1,
        )
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        batch_size,
        query_length,
        self.num_heads,
        self.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size,
        query_length,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size,
        query_length,
        self.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    cache_length = 0
    if past_key_value is not None:
        if not isinstance(past_key_value, LiquidFusionCache):
            raise TypeError("LiquidFusion attention requires LiquidFusionCache")
        if self.layer_idx is None:
            raise ValueError("LiquidFusion attention requires a layer index")
        cache_length = past_key_value.get_usable_length(query_length, self.layer_idx)

    physical_length = cache_length + query_length
    absolute_length = (
        int(position_ids.max().item()) + 1
        if position_ids is not None
        else physical_length
    )
    cos, sin = self.rotary_emb(
        value_states,
        seq_len=max(physical_length, absolute_length),
    )
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            {
                "sin": sin,
                "cos": cos,
                "position_ids": position_ids,
            },
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    key_length = key_states.shape[-2]

    attention_logits = torch.matmul(
        query_states,
        key_states.transpose(2, 3),
    ) / math.sqrt(self.head_dim)

    expected_shape = (
        batch_size,
        self.num_heads,
        query_length,
        key_length,
    )
    if attention_logits.shape != expected_shape:
        raise ValueError(
            f"Attention weights must have shape {expected_shape}, got {tuple(attention_logits.shape)}"
        )

    if attention_mask is not None and query_length > 1:
        causal_mask = attention_mask[
            :,
            :,
            :query_length,
            :key_length,
        ]
        attention_logits = attention_logits + causal_mask

    attention_probs = nn.functional.softmax(
        attention_logits,
        dim=-1,
        dtype=torch.float32,
    ).to(query_states.dtype)

    if past_key_value is not None:
        past_key_value.record_attention(self.layer_idx, attention_probs)

    attention_output = torch.matmul(
        nn.functional.dropout(
            attention_probs,
            p=self.attention_dropout,
            training=self.training,
        ),
        value_states,
    )

    expected_output_shape = (
        batch_size,
        self.num_heads,
        query_length,
        self.head_dim,
    )
    if attention_output.shape != expected_output_shape:
        raise ValueError(
            f"Attention output must have shape {expected_output_shape}, got {tuple(attention_output.shape)}"
        )

    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(
        batch_size,
        query_length,
        self.hidden_size,
    )

    if self.config.pretraining_tp > 1:
        attention_output = attention_output.split(
            self.hidden_size // self.config.pretraining_tp,
            dim=2,
        )
        output_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp,
            dim=1,
        )
        attention_output = sum(
            F.linear(attention_output[index], output_slices[index])
            for index in range(self.config.pretraining_tp)
        )
    else:
        attention_output = self.o_proj(attention_output)

    if not output_attentions:
        attention_probs = None

    return attention_output, attention_probs, past_key_value


def liquid_fusion_model_forward(self, *args, **kwargs):
    use_cache = kwargs.get("use_cache")
    if use_cache is None:
        use_cache = self.config.use_cache
    past_key_values = kwargs.get("past_key_values")
    if use_cache and past_key_values is None:
        kwargs["past_key_values"] = LiquidFusionCache(
            sink_size=self._liquid_fusion_sink_size,
            heavy_budget=self._liquid_fusion_heavy_budget,
            recent_budget=self._liquid_fusion_recent_budget,
        )
    elif past_key_values is not None and not isinstance(
        past_key_values,
        LiquidFusionCache,
    ):
        raise TypeError("LiquidFusion does not accept legacy or unrelated cache types")

    attention_mask = kwargs.get("attention_mask")
    if (
        attention_mask is not None
        and attention_mask.ndim == 2
        and torch.any(attention_mask == 0)
    ):
        raise ValueError("LiquidFusion currently requires unpadded batches")

    if isinstance(past_key_values, LiquidFusionCache) and past_key_values.seen_tokens:
        input_tensor = kwargs.get("input_ids")
        if input_tensor is None:
            input_tensor = kwargs.get("inputs_embeds")
        query_length = input_tensor.shape[1]
        if kwargs.get("position_ids") is None:
            kwargs["position_ids"] = torch.arange(
                past_key_values.seen_tokens,
                past_key_values.seen_tokens + query_length,
                dtype=torch.long,
                device=input_tensor.device,
            ).unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 2:
            kwargs["attention_mask"] = torch.ones(
                input_tensor.shape[0],
                past_key_values.get_seq_length() + query_length,
                dtype=attention_mask.dtype,
                device=input_tensor.device,
            )

    return self._liquid_fusion_original_forward(*args, **kwargs)


def liquid_fusion_prepare_inputs(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    use_cache = kwargs.get("use_cache")
    if use_cache is not False and past_key_values is None:
        past_key_values = LiquidFusionCache(
            sink_size=self._liquid_fusion_sink_size,
            heavy_budget=self._liquid_fusion_heavy_budget,
            recent_budget=self._liquid_fusion_recent_budget,
        )
    if past_key_values is not None and not isinstance(
        past_key_values,
        LiquidFusionCache,
    ):
        raise TypeError("LiquidFusion generation requires LiquidFusionCache")
    if attention_mask is not None and torch.any(attention_mask == 0):
        raise ValueError("LiquidFusion currently requires unpadded batches")

    seen_tokens = past_key_values.seen_tokens if past_key_values is not None else 0
    if seen_tokens:
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            unprocessed = attention_mask.shape[1] - seen_tokens
            input_ids = input_ids[:, -unprocessed:]
        elif seen_tokens < input_ids.shape[1]:
            input_ids = input_ids[:, seen_tokens:]

    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            if seen_tokens:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = torch.arange(
                seen_tokens,
                seen_tokens + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)

    if seen_tokens:
        attention_mask = torch.ones(
            input_ids.shape[0],
            past_key_values.get_seq_length() + input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )

    if inputs_embeds is not None and not seen_tokens:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def convert_to_liquid_fusion(
    model,
    sink_size: int = 4,
    heavy_budget: int = 50,
    recent_budget: int = 50,
):
    if getattr(model, "_liquid_fusion_enabled", False):
        return model
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("LiquidFusion currently supports LlamaForCausalLM-compatible models")

    LiquidFusionCache(sink_size, heavy_budget, recent_budget)
    model.config._attn_implementation_internal = "eager"

    for layer in model.model.layers:
        if not isinstance(layer.self_attn, LlamaAttention):
            raise TypeError("LiquidFusion requires eager LlamaAttention layers")
        layer.self_attn.forward = MethodType(
            liquid_fusion_forward,
            layer.self_attn,
        )

    model.model._liquid_fusion_sink_size = sink_size
    model.model._liquid_fusion_heavy_budget = heavy_budget
    model.model._liquid_fusion_recent_budget = recent_budget
    model.model._liquid_fusion_original_forward = model.model.forward
    model.model.forward = MethodType(liquid_fusion_model_forward, model.model)

    model._liquid_fusion_sink_size = sink_size
    model._liquid_fusion_heavy_budget = heavy_budget
    model._liquid_fusion_recent_budget = recent_budget
    model.prepare_inputs_for_generation = MethodType(
        liquid_fusion_prepare_inputs,
        model,
    )
    model._liquid_fusion_enabled = True
    return model