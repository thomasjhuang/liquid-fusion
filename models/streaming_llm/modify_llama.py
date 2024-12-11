# coding=utf-8
# Align with transformer==4.36.2
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch import Tensor

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward, logging

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
import contextlib

from transformers import LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb, rotate_half

from types import MethodType
from MoA.attention.density_calculation import streamingllm_attention_density, streamingllm_kv_cache_density

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)

    return x_embed

"""
streamingLLM implementation
"""

def LlamaModel_Streamingllm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        # use_legacy_cache = not isinstance(past_key_values, Cache)
        # if use_legacy_cache:
        if past_key_values is None:
            past_key_values = StreamingllmDynamicCache()
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    ## changes ###
    if use_cache:
        next_cache = next_decoder_cache
    ### end ###
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    ### !you can just past the last hidden states in generation mode
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states[:, -1:, :],
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def LlamaAttention_streamingllm_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        is_decode = (q_len == 1)
        assert bsz == 1, "only support batch size 1 now"

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # ## the key inside the kv cache does not contain position embedding here
        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        band_size = self.band_size
        global_size = self.global_size

        if is_decode:
            ## the key inside the kv cache does not contain position embedding here
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs, band_size=band_size, global_size=global_size)
            
            kv_seq_len = key_states.shape[2]
            
            # create position_ids again, using kv_seq_len, possibly with batch_size
            position_ids = torch.tensor(kv_seq_len-1, dtype=torch.long, device=position_ids.device).unsqueeze(0).unsqueeze(0)

            ### add position embedding ###
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)

            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
            ### end of position embedding ###

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # attention_mask = None

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        
        else:
            # the key inside the kv cache does not contain position embedding here
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            if key_states.shape[2] > band_size + global_size:
                concatenated_key_states = torch.cat([key_states[:, :, :global_size, :], key_states[:, :, -band_size:, :]], dim=2)
                concatenated_value_states = torch.cat([value_states[:, :, :global_size, :], value_states[:, :, -band_size:, :]], dim=2)
                past_key_value.update(concatenated_key_states, concatenated_value_states, self.layer_idx, cache_kwargs)
            else:
                past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)



            # save a copy of original states
            original_query_states = query_states.clone().detach()
            original_key_states = key_states.clone().detach()

            # first do normal attention, which calculates the global_size area
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # we will add causal mask after the attention weights are calculated
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # assert attention_mask is not None, "causal mask is required for streaming attention"

            if band_size + global_size < kv_seq_len:
                # further masking
                attn_weights  = attn_weights + self.global_mask[:, :, :kv_seq_len, :kv_seq_len]

                # rope for key
                new_key_states = original_key_states[:, :, :global_size, :]
                new_query_states = original_query_states[:, :, band_size + global_size:, :]

                new_key_position_ids = torch.arange(global_size, device=position_ids.device).unsqueeze(0)
                new_key_states = apply_rotary_pos_emb_single(new_key_states, cos, sin, new_key_position_ids)

                # rope for query
                new_query_position_ids = torch.full((1, kv_seq_len - band_size - global_size), band_size + global_size - 1, device=position_ids.device)
                new_query_states = apply_rotary_pos_emb_single(new_query_states, cos, sin, new_query_position_ids)

                # repeat k/v heads if n_kv_heads < n_heads
                new_key_states = repeat_kv(new_key_states, self.num_key_value_groups)
                new_query_states = repeat_kv(new_query_states, self.num_key_value_groups)

                new_attn_weights = torch.matmul(new_query_states, new_key_states.transpose(2, 3)) / math.sqrt(
                    self.head_dim
                )

                # copy new_attn_weights to attn_weights
                attn_weights[:, :, band_size + global_size:, :global_size] = new_attn_weights[:, :, :, :]


            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            # upcast attention to fp32
            if kv_seq_len < 12000:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            else:
                attn_weights[:, :self.num_heads // 2, :, :] = nn.functional.softmax(attn_weights[:, :self.num_heads // 2, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights[:, self.num_heads // 2:, :, :] = nn.functional.softmax(attn_weights[:, self.num_heads // 2:, :, :], dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def create_streaming_attention_mask(token_len, global_size, band_size):
    # Start by creating an empty mask filled with False (0)
    mask = torch.zeros(1, 1, token_len, token_len, dtype=torch.bool)
    
    # Apply causal mask with band pattern
    for i in range(token_len):
        # Ensuring causality and the band pattern
        start = max(i - band_size + 1, 0)
        end = min(i + 1, token_len)  # Causal limit
        mask[:, :, i, start:end] = True
    
    # Overwrite the first `global_size` columns to True, only up to the diagonal for causality
    if global_size > 0:
        for i in range(token_len):
            mask[:, :, i, :min(global_size, i+1)] = True
        
    return mask


def LlamaModel_use_streamingllm_attention(model, global_size, band_size, device='cuda', max_length=16384):
    """
    Set the model instance to use streamingllm like attention instead of llama attention
    """

    # create a mask in advance
    attention_mask = create_streaming_attention_mask(max_length, global_size, band_size)
    attention_mask = (~attention_mask).to(torch.float16) * torch.finfo(torch.float16).min

    # get the device name of all layers. they may be on different cuda devices
    device_list = set([str(layer.self_attn.o_proj.weight.device) for layer in model.layers])
    attention_mask_dict = {device: attention_mask.to(device) for device in device_list}
    print(f"parameters are on devices: {device_list}")

    for layer in model.layers:
        layer.self_attn.forward = MethodType(LlamaAttention_streamingllm_forward, layer.self_attn)
        layer.self_attn.global_size = global_size
        layer.self_attn.band_size = band_size
        layer.self_attn.global_mask = attention_mask_dict[str(layer.self_attn.o_proj.weight.device)]

    attention_density = streamingllm_attention_density(global_size, band_size, max_length)
    kv_cache_deisnty = streamingllm_kv_cache_density(global_size, band_size, max_length)

    print(f"streamingllm: global_size = {global_size}, band_size = {band_size}, max_length = {max_length}\n attention_density: {attention_density}\n kv_cache_density: {kv_cache_deisnty}")

    model.forward = MethodType(LlamaModel_Streamingllm_forward, model)


class StreamingllmDynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __getitem__(self, layer_idx: int) -> List[Tuple[Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        update_seen_tokens: Optional[int] = None,
        global_size: Optional[int] = None,
        band_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`Tensor`):
                The new key states to cache.
            value_states (`Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            if update_seen_tokens is not None:
                self.seen_tokens += update_seen_tokens
            else:
                self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
            if (
                band_size is not None
                and global_size is not None
                and (band_size + global_size < self.key_cache[layer_idx].shape[2])
            ):
                self.key_cache[layer_idx] = torch.cat(
                    [
                        self.key_cache[layer_idx][:, :, :global_size, :],
                        self.key_cache[layer_idx][:, :, -band_size:, :],
                    ],
                    dim=-2,
                )
                self.value_cache[layer_idx] = torch.cat(
                    [
                        self.value_cache[layer_idx][:, :, :global_size, :],
                        self.value_cache[layer_idx][:, :, -band_size:, :],
                    ],
                    dim=-2,
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )