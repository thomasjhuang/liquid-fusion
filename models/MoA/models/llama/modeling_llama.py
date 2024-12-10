# coding=utf-8
# Align with transformer==4.36.2
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union, Dict
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
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb, rotate_half

from types import MethodType

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

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
    **kwargs
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
                past_key_value.update(concatenated_key_states, concatenated_value_states, self.layer_idx, cache_kwargs, update_cache_position=key_states.shape[2])
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


def streamingllm_attention_density(
    global_size: int = 4,
    band_size: int = 1024,
    kv_seq_len: int = 8192,
):  
    num_total = 0
    num_attended = 0

    for i in range(kv_seq_len):
        for j in range(kv_seq_len):
            if i < j:
                continue
            num_total += 1

            if (j < global_size) or (i - j < band_size):
                num_attended += 1
    
    return num_attended / num_total

def streamingllm_kv_cache_density(
    global_size: int = 4,
    band_size: int = 1024,
    kv_seq_len: int = 8192,
):
    if global_size + band_size > kv_seq_len:
        return 1.0

    else:
        return (global_size + band_size) / kv_seq_len


def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)

    return x_embed


from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache as OriginalCache
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
import math
from torch import Tensor

class NewDynamicCache(Cache):
    def __init__(
        self,
        pattern_num: List[int],
        global_size: List[List[int]],
        band_size: List[List[int]],
        pattern_index: List[List[int]],
    ) -> None:
        self.cache_position = 0
        self.layer_cache = []
        for i in range(len(pattern_num)):
            self.layer_cache.append(
                LayerCache(
                    pattern_num[i], global_size[i], band_size[i], pattern_index[i]
                )
            )

    def __len__(self):
        return len(self.layer_cache)

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self.cache_position += key_states.shape[-2]

        # Update the cache
        key_cache, value_cache = self.layer_cache[layer_idx].update(
            key_states, value_states
        )

        return key_cache, value_cache

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        raise NotImplementedError

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            raise NotImplementedError(
                "Reordering the cache is not implemented currently!"
            )
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )


class StreamingllmDynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.cache_position = (
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
        update_cache_position: Optional[int] = None,
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
            if update_cache_position is not None:
                self.cache_position += update_cache_position
            else:
                self.cache_position += key_states.shape[-2]

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


class LayerCache(Cache):
    def __init__(
        self,
        pattern_num: int,
        global_size: List[int],
        band_size: List[int],
        pattern_index: List[int],
    ) -> None:
        self.pattern_num = pattern_num
        self.global_size = global_size
        self.band_size = band_size
        self.pattern_index = pattern_index
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []

        self.replace_index = [self.global_size[i] for i in range(self.pattern_num)]

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        key_states, value_states : [bz, num_heads, seq_len, head_dim]
        currently, after the first insert, the size of key_cache and value_cache will be fixed
        """
        if len(self.key_cache) == 0:
            # only keep the first self.global_size and the last self.band_size of seq_len dimension
            # if key_states.size(2) < self.global_size + self.band_size:
            #     raise ValueError(f"seq_len of key_states should be at least {self.global_size + self.band_size}")
            seperated_key_states = self.seperate_states(key_states)
            seperated_value_states = self.seperate_states(value_states)

            for i in range(self.pattern_num):
                self.key_cache.append(
                    torch.cat(
                        [
                            seperated_key_states[i][:, :, : self.global_size[i], :],
                            seperated_key_states[i][:, :, -self.band_size[i] :, :],
                        ],
                        dim=2,
                    )
                )
                self.value_cache.append(
                    torch.cat(
                        [
                            seperated_value_states[i][:, :, : self.global_size[i], :],
                            seperated_value_states[i][:, :, -self.band_size[i] :, :],
                        ],
                        dim=2,
                    )
                )

                # whe the first insert, return the original key_states and value_states for prefilling
                return key_states, value_states

        else:
            assert (
                key_states.size(2) == 1
            ), "in decoding mode, key_states should only have one token"
            seperated_key_states = self.seperate_states(key_states)
            seperated_value_states = self.seperate_states(value_states)
            # print(f"replace_index: {self.replace_index}")
            for i in range(self.pattern_num):
                replace_index = self.replace_index[i]
                self.key_cache[i][:, :, replace_index : replace_index + 1, :] = (
                    seperated_key_states[i]
                )
                self.value_cache[i][:, :, replace_index : replace_index + 1, :] = (
                    seperated_value_states[i]
                )

                if replace_index == self.global_size[i] + self.band_size[i] - 1:
                    self.replace_index[i] = self.global_size[i]
                else:
                    self.replace_index[i] += 1

            return self.key_cache, self.value_cache

    def seperate_states(self, states: Tensor) -> List[Tensor]:
        """
        states: [bz, num_heads, seq_len, head_dim]
        """
        result = []

        for i in range(self.pattern_num):
            result.append(
                states[:, self.pattern_index[i] : self.pattern_index[i + 1], :, :]
            )

        return result


class CircularCacheSingle(Cache):
    def __init__(
        self,
    ) -> None:
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.cache_position = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen\
        )
        self.global_size = []
        self.band_size = []
        self.replace_index = []
        self.kv_len = []

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
        global_size: Optional[int] = None,
        band_size: Optional[int] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
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
                Additional arguments for the cache subclass. No additional arguments are used in `CircularSingleCache`.
            global_size (`int`, `optional`):
                The size of the global part of the this layer head pattern.
            band_size (`int`, `optional`):
                The size of the band part of the this layer head pattern.

        Return:
            A tuple containing the updated key and value states.
        """

        # # Update the number of seen tokens
        # if layer_idx == 0 and key_states is not None:
        #     self.seen_tokens += key_states.shape[-2]

        # # Update the cache
        # if len(self.key_cache) <= layer_idx:

        #     if key_states is None:
        #         self.key_cache.append(None)
        #         self.value_cache.append(None)
        #         self.global_size.append(None)
        #         self.band_size.append(None)
        #         self.replace_index.append(None)
        #         return None, None

        #     assert global_size is not None and band_size is not None, "In the first update, global_size and band_size must be provided"

        #     self.global_size.append(global_size)
        #     self.band_size.append(band_size)

        #     assert len(self.key_cache) == layer_idx
        #     seq_len = key_states.shape[2]
        #     if seq_len >= global_size + band_size:
        #         self.key_cache.append(torch.cat([key_states[:, :, :global_size, :], key_states[:, :, -band_size:, :]], dim=2))
        #         self.value_cache.append(torch.cat([value_states[:, :, :global_size, :], value_states[:, :, -band_size:, :]], dim=2))
        #     else:
        #         # when the seq len is too short, just use the original key_states and value_states'
        #         # update global size
        #         # ! this shortens the band_size
        #         self.key_cache.append(key_states)
        #         self.value_cache.append(value_states)

        #         self.global_size[layer_idx] = seq_len - band_size

        #     if seq_len < global_size:
        #         raise ValueError(f"seq_len of key_states should be at least {global_size}")

        #     self.replace_index.append(global_size)
        # else:

        #     if key_states is None:
        #         return None, None

        #     assert key_states.size(2) == 1, "in decoding mode, key_states should only have one token"

        #     replace_index = self.replace_index[layer_idx]
        #     self.key_cache[layer_idx][:, :, replace_index:replace_index+1, :] = key_states
        #     self.value_cache[layer_idx][:, :, replace_index:replace_index+1, :] = value_states

        #     if replace_index == self.global_size[layer_idx] + self.band_size[layer_idx] - 1:
        #         self.replace_index[layer_idx] = self.global_size[layer_idx]
        #     else:
        #         self.replace_index[layer_idx] += 1

        # return self.key_cache[layer_idx], self.value_cache[layer_idx]

        ### new implementation ###

        # Update the number of seen tokens
        if layer_idx == 0 and key_states is not None:
            self.cache_position += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # initialize the cache

            if key_states is None:
                self.key_cache.append(None)
                self.value_cache.append(None)
                self.global_size.append(None)
                self.band_size.append(None)
                self.replace_index.append(None)
                self.kv_len.append(None)
                return None, None

            assert (
                global_size is not None and band_size is not None
            ), "In the first update, global_size and band_size must be provided"

            self.global_size.append(global_size)
            self.band_size.append(band_size)

            assert len(self.key_cache) == layer_idx
            seq_len = key_states.shape[2]
            if seq_len >= global_size + band_size:

                self.key_cache.append(
                    torch.cat(
                        [
                            key_states[:, :, :global_size, :],
                            key_states[:, :, -band_size:, :],
                        ],
                        dim=2,
                    )
                )
                self.value_cache.append(
                    torch.cat(
                        [
                            value_states[:, :, :global_size, :],
                            value_states[:, :, -band_size:, :],
                        ],
                        dim=2,
                    )
                )

                self.kv_len.append(global_size + band_size)
                self.replace_index.append(global_size)

            else:
                # preallocate global_size + band_size space
                longer_allocated_key = torch.empty(
                    key_states.shape[0],
                    key_states.shape[1],
                    global_size + band_size,
                    key_states.shape[3],
                    device=key_states.device,
                    dtype=key_states.dtype,
                )
                longer_allocated_value = torch.empty(
                    value_states.shape[0],
                    value_states.shape[1],
                    global_size + band_size,
                    value_states.shape[3],
                    device=value_states.device,
                    dtype=value_states.dtype,
                )
                longer_allocated_key[:, :, :seq_len, :] = key_states
                longer_allocated_value[:, :, :seq_len, :] = value_states
                self.key_cache.append(longer_allocated_key)
                self.value_cache.append(longer_allocated_value)

                self.kv_len.append(seq_len)
                self.replace_index.append(seq_len)

            if seq_len < global_size:
                raise ValueError(
                    f"seq_len of key_states should be at least {global_size}"
                )

        else:

            if key_states is None:
                return None, None

            assert (
                key_states.size(2) == 1
            ), "in decoding mode, key_states should only have one token"

            replace_index = self.replace_index[layer_idx]
            self.key_cache[layer_idx][
                :, :, replace_index : replace_index + 1, :
            ] = key_states
            self.value_cache[layer_idx][
                :, :, replace_index : replace_index + 1, :
            ] = value_states

            if (
                replace_index
                == self.global_size[layer_idx] + self.band_size[layer_idx] - 1
            ):
                self.replace_index[layer_idx] = self.global_size[layer_idx]
            else:
                self.replace_index[layer_idx] += 1

            if (
                self.kv_len[layer_idx]
                < self.global_size[layer_idx] + self.band_size[layer_idx]
            ):
                self.kv_len[layer_idx] += 1

        if (
            self.kv_len[layer_idx]
            == self.global_size[layer_idx] + self.band_size[layer_idx]
        ):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            return (
                self.key_cache[layer_idx][:, :, : self.kv_len[layer_idx], :],
                self.value_cache[layer_idx][:, :, : self.kv_len[layer_idx], :],
            )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # if len(self.key_cache) <= layer_idx:
        #     return 0
        # return self.key_cache[layer_idx].shape[-2]

        ### new implementation ###
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.kv_len[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Reordering the cache is not implemented currently!")


class CircularCache(Cache):
    def __init__(self, pattern_num: int = 1, num_layers=32) -> None:
        self.cache = []
        for _ in range(pattern_num):
            self.cache.append(CircularCacheSingle())

        self.num_layers = num_layers

        # self.seen_tokens = 0
        self.cache_position = 0
        self.current_tokens = 0

    def __len__(self):

        return len(self.cache[0])

    def update(
        self,
        key_states: List[Tensor],
        value_states: List[Tensor],
        layer_idx: int,
        global_size: Optional[List[int]] = None,
        band_size: Optional[List[int]] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert layer_idx < self.num_layers

        if layer_idx == 0:
            self.cache_position += key_states[0].shape[-2]
        if layer_idx == self.num_layers - 1:
            self.current_tokens = self.cache_position

        if global_size is None:
            global_size = [None for _ in range(len(key_states))]
        if band_size is None:
            band_size = [None for _ in range(len(key_states))]

        assert len(key_states) == len(self.cache)
        assert len(value_states) == len(self.cache)
        assert len(global_size) == len(self.cache)
        assert len(band_size) == len(self.cache)

        updated_key_states = []
        updated_value_states = []

        for i in range(len(key_states)):
            key_state, value_state = self.cache[i].update(
                key_states[i],
                value_states[i],
                layer_idx,
                global_size[i],
                band_size[i],
                cache_kwargs,
            )
            updated_key_states.append(key_state)
            updated_value_states.append(value_state)

        return updated_key_states, updated_value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # ! warning: this function is meaningless, just a place holder
        return self.cache[0].get_seq_length(layer_idx)
        # raise NotImplementedError

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Reordering the cache is not implemented currently!")

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        # return self.get_seq_length(layer_idx)
        raise NotImplementedError


class StaticCircularCache(Cache):
    """
    A cache that do not grow dynamically as more tokens are generated. This is the default for mixture-of-sparse-attention (MoA) method.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`. Note that when using the `update` function, the shape of each layer is `[batch_size, num_heads, seq_len, head_dim]`.
    Each head has its own total cache size consisting two parts: static and circular. The static part is the first `global_size` tokens and the circular part is the last `band_size` tokens. When updating the cache, the circular part will be replaced by the new tokens, while the static part will be kept unchanged.
    """

    def __init__(
        self,
        cache_size: List[List[int]],
        batch_size: int = 1,
        head_dim: int = 128,
        static_size: Optional[List[List[int]]] = None,
        device: torch.device = "cpu",
        dtype: torch.dtype = None,
        update_cache_content: bool = True,
    ) -> None:
        """
        Parameters:
            cache_size (`List[List[int]]`):
                The total cache size for each head in each layer. The cache size is the sum of the global part and the band part.
            batch_size (*optional*, `int`):
                The batch size of the input data. If not provided, it will be set to 1.
            head_dim (*optional*, `int`):
                The dimension of the head. If not provided, it will be set to 64.
            static_size (*optional*, `List[List[int]]`):
                The size of the static part for each head in each layer. If not provided, the static part will be set to 0.
            device (*optional*, `torch.device`):
                The device of the cache. If not provided, it will be set to `cpu`.
            dtype (*optional*, `torch.dtype`):
                The data type of the cache. If not provided, it will be set to `torch.float16`.
            update_cache_content (*optional*, `bool`):
                Whether to update the cache content when calling `.update`. If not provided, it will be set to `True`. Set to False if the kernel update the content
        """
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.float16
        self.device = device
        self.update_cache_content = update_cache_content
        self.batch_size = batch_size
        self.head_dim = head_dim

        # initialize the cache sizes and indices
        self.num_layers = len(cache_size)
        self.num_head_for_each_layer = [
            len(cache_size[layer_id]) for layer_id in range(self.num_layers)
        ]
        if static_size is None:
            static_size = [
                [0 for _ in range(len(cache_size[layer_id]))]
                for layer_id in range(self.num_layers)
            ]

        self.cache_max_length: List[torch.LongTensor] = [
            torch.tensor(cache_size_this_layer, device=device, dtype=torch.int64)
            for cache_size_this_layer in cache_size
        ]
        self.static_cache_size: List[torch.LongTensor] = [
            torch.tensor(static_size_this_layer, device=device, dtype=torch.int64)
            for static_size_this_layer in static_size
        ]
        self.circular_cache_size: List[torch.LongTensor] = [
            torch.tensor(
                cache_size_this_layer - static_size_this_layer,
                device=device,
                dtype=torch.int64,
            )
            for cache_size_this_layer, static_size_this_layer in zip(
                self.cache_max_length, self.static_cache_size
            )
        ]

        self.layer_cache_size: List[int] = [
            sum(this_cache_size) for this_cache_size in cache_size
        ]  # total cache size for each layer
        head_start_index = [
            [] for layer_id in range(self.num_layers)
        ]  # the starting index of each head in each layer

        for layer_id in range(self.num_layers):
            for head_id in range(len(cache_size[layer_id])):
                if head_id == 0:
                    head_start_index[layer_id].append(0)
                else:
                    head_start_index[layer_id].append(
                        head_start_index[layer_id][-1]
                        + cache_size[layer_id][head_id - 1]
                    )
        self.cache_head_start_index = [
            torch.tensor(
                head_start_index[layer_id], device=self.device, dtype=torch.int64
            )[None, :]
            .expand(batch_size, self.num_head_for_each_layer[layer_id])
            .contiguous()
            for layer_id in range(self.num_layers)
        ]  # shape: (batch_size, num_heads) * num_layers, the starting index of each head in each layer

        for layer_id in range(self.num_layers):
            # add another start index to indicate the end of the last head
            head_start_index[layer_id].append(
                head_start_index[layer_id][-1] + cache_size[layer_id][-1]
            )
        self.head_index = [
            torch.tensor(head_start_index[layer_id], dtype=torch.int64, device=device)
            for layer_id in range(self.num_layers)
        ]  # shape: (num_heads+1) * num_layers, the starting index of each head in each layer; contains the additional index to show the end of the cache

        self.circular_cache_head_index: List[torch.LongTensor] = [
            torch.tensor(
                [
                    head_start_index[layer_id][head_id] + static_size[layer_id][head_id]
                    for head_id in range(self.num_head_for_each_layer[layer_id])
                ],
                device=device,
                dtype=torch.int64,
            )
            for layer_id in range(self.num_layers)
        ]  # the starting index of the circular part of each head in each layer

        self.cache_update_index: List[torch.LongTensor] = [
            torch.tensor(
                head_start_index[layer_id][:-1], dtype=torch.int64, device=device
            ).expand(batch_size, -1)
            for layer_id in range(self.num_layers)
        ]  # initialize the update index to the beginning of the cache, it will later circulate within the circular part after filling the cache, shape (batch_size, num_heads) * num_layers

        # parameter check
        assert len(static_size) == self.num_layers
        for layer_id in range(self.num_layers):
            assert len(static_size[layer_id]) == len(cache_size[layer_id])

        # initialize the cache
        self.cache_position = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self._kv_len = [
            0 for _ in range(self.num_layers)
        ]  # the length of the key and value cache for each layer

        # initialize as meta tensor to avoid extensive space on single gpu for multi-gpu inference
        self.key_cache: List[Tensor] = [
            torch.zeros(
                batch_size,
                total_cache_size_this,
                head_dim,
                dtype=self.dtype,
                device='meta',
            )
            for total_cache_size_this in self.layer_cache_size
        ]
        self.value_cache: List[Tensor] = [
            torch.zeros(
                batch_size,
                total_cache_size_this,
                head_dim,
                dtype=self.dtype,
                device='meta',
            )
            for total_cache_size_this in self.layer_cache_size
        ]
        self.mask_cache: List[torch.LongTensor] = [
            torch.zeros(
                batch_size, total_cache_size_this, dtype=torch.int64, device=device
            )
            for total_cache_size_this in self.layer_cache_size
        ]  # 1 means not masked, 0 means masked
        self.cache_valid_length: List[torch.LongTensor] = [
            torch.zeros(
                (batch_size, self.num_head_for_each_layer[layer_id]),
                dtype=torch.int64,
                device=device,
            )
            for layer_id in range(self.num_layers)
        ]

    @staticmethod
    def head_start_index_valid_length_to_head_index(
        head_start_index: Tensor,
        head_valid_length: Tensor,
    ) -> Tensor:
        """
        Convert the starting index and valid length of each head to the head index.
        """
        assert (head_start_index == head_start_index[0]).all().item()
        assert (head_valid_length == head_valid_length[0]).all().item()
        head_index = torch.cat((head_start_index[0], (head_start_index[0][-1]+head_valid_length[0][-1]).reshape(-1))) # head_index = past_key_value.head_index[self.layer_idx]
        return head_index.contiguous()

    @staticmethod
    def head_index_to_head_start_index_valid_length(
        head_index: Tensor,
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert the head index to the starting index and valid length of each head.
        """
        # valid length is more complex than head_index
        raise NotImplementedError
    
        # Calculate head_start_index from head_index
        head_start_index = head_index[:-1]

        # Calculate head_valid_length from head_index
        # The length of each segment is the difference between consecutive entries in head_index
        head_valid_length = head_index[1:] - head_index[:-1]

        # expand by batch size
        head_start_index = head_start_index[None, :].expand(batch_size, -1)
        head_valid_length = head_valid_length[None, :].expand(batch_size, -1)

        return head_start_index.contiguous(), head_valid_length.contiguous()


    @staticmethod
    def to_uncontigious(
        tensor: Tensor,
        head_index: Tensor,
    ) -> List[Tensor]:
        """
        Split the tensor to each head according to the head_index

        Parameters:
            tensor (`Tensor`):
                The expected shape for each tensor is `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`.
            head_index (`Tensor`):
                The starting index of each head, shape (num_heads+1).

        Return:
            A list of tensors, each tensor is the cache for each head of shape `[batch_size, cache_size_of_head_h, head_dim]`.
        """
        return [
            tensor[:, head_index[head_id] : head_index[head_id + 1], :]
            for head_id in range(len(head_index) - 1)
        ]

    @staticmethod
    def to_group_contigious(
        tensor: Tensor,
        head_index: Tensor,
    ) -> List[Tensor]:
        """
        Split the tensor into each group, where heads within the group share the same cache size.

        Parameters:
            tensor (`Tensor`):
                The expected shape for each tensor is either `[batch_size, \sum_h^H cache_size_of_head_h, head_dim]`
                or `[batch_size, \sum_h^H cache_size_of_head_h]`.
            head_index (`Tensor`):
                The starting index of each head, shape (num_heads+1).

        Return:
            A list of tensors, each tensor is the cache for each group. The shape of each tensor will be
            `[batch_size, num_heads_in_group, cache_size_of_head, head_dim]` if head_dim is present,
            otherwise `[batch_size, num_heads_in_group, cache_size_of_head]`.
        """
        groups = []
        batch_size = tensor.shape[0]
        has_head_dim = len(tensor.shape) == 3
        if has_head_dim:
            head_dim = tensor.shape[2]

        cache_size = head_index[1:] - head_index[:-1]  # shape: (num_heads)

        # Identify unique consecutive cache sizes and their boundaries
        unique_sizes, inverse_indices, counts = torch.unique_consecutive(
            cache_size, return_inverse=True, return_counts=True
        )

        # Prepare to group
        current_idx = 0
        for size, count in zip(unique_sizes, counts):
            # Start and end indices in head_index
            group_start_index = current_idx
            group_end_index = group_start_index + count

            # Slicing tensor according to the computed start and end
            start = head_index[group_start_index].item()
            end = head_index[group_end_index - 1].item() + size.item()

            # Extract the relevant slice from the tensor
            if has_head_dim:
                group_tensor = tensor[:, start:end, :]
                # Reshape to add the head dimension: [batch_size, num_heads_in_group, cache_size_of_head, head_dim]
                group_tensor = group_tensor.view(
                    batch_size, count, size.item(), head_dim
                )
            else:
                group_tensor = tensor[:, start:end]
                # Reshape to add the head dimension: [batch_size, num_heads_in_group, cache_size_of_head]
                group_tensor = group_tensor.view(batch_size, count, size.item())

            # Append to groups
            groups.append(group_tensor.contiguous())

            # Update the current index
            current_idx = group_end_index

        return groups

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        # position_ids: Tensor, # can be used to move the sink part to the left side
        attention_mask: Optional[torch.IntTensor] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters:
            key_states (`Tensor`):
                The expected shapes for each key_states or value_states are
                    `[batch_size, num_heads, seq_len, head_dim]`.
            value_states (`Tensor`):
                The expected shapes for each key_states or value_states are
                    `[batch_size, num_heads, seq_len, head_dim]`.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            attention_mask (`torch.BoolTensor`, `optional`):
                The attention mask to apply to the cache. If not provided, no mask will be applied. The mask is a int64 tensor, where 1 means preverse the value and 0 means masked. The expected shape is `[batch_size, seq_len]`.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `StaticCircularCache`.
        Return:
            A tuple containing the updated key and value states. The expected shapes for each key_states or value_states are `[batch_size, \sum_h^H cache_size_of_head_h , head_dim]`.
        """
        assert layer_idx < self.num_layers
        if key_states is not None:
            assert key_states.shape[1] == self.num_head_for_each_layer[layer_idx]

        # move the tensors to correct devices for multi-gpu inference
        device = key_states.device
        assert value_states.device == device, "key_states and value_states should be on the same device"
        if attention_mask is not None:
            assert attention_mask.device == device, "attention_mask should be on the same device as key_states"
        
        # instantiate meta tensor to the correct device
        if self.key_cache[layer_idx].device == torch.device('meta'):
            self.key_cache[layer_idx] = torch.zeros_like(self.key_cache[layer_idx], device=device)
        if self.value_cache[layer_idx].device == torch.device('meta'):
            self.value_cache[layer_idx] = torch.zeros_like(self.value_cache[layer_idx], device=device)
        self.mask_cache[layer_idx] = self.mask_cache[layer_idx].to(device=device)
        self.cache_update_index[layer_idx] = self.cache_update_index[layer_idx].to(device=device)
        self.cache_head_start_index[layer_idx] = self.cache_head_start_index[layer_idx].to(device=device)
        self.circular_cache_head_index[layer_idx] = self.circular_cache_head_index[layer_idx].to(device=device)
        self.cache_valid_length[layer_idx] = self.cache_valid_length[layer_idx].to(device=device)
        self.cache_max_length[layer_idx] = self.cache_max_length[layer_idx].to(device=device)
        self.circular_cache_size[layer_idx] = self.circular_cache_size[layer_idx].to(device=device)
        self.static_cache_size[layer_idx] = self.static_cache_size[layer_idx].to(device=device)

        # if do not update cache content, key and value may be None
        if self.update_cache_content:
            batch_size = key_states.shape[0]
            seq_len = key_states.shape[-2]
            head_dim = key_states.shape[-1]
        else:
            batch_size = self.batch_size
            seq_len = key_states.shape[-2] if key_states is not None else 1
            head_dim = self.head_dim
        num_head = self.num_head_for_each_layer[layer_idx]

        # Update the number of seen tokens
        if layer_idx == 0:
            self.cache_position += seq_len

        is_decode = seq_len == 1

        if is_decode:
            # only update the last token of the cache, use simpler logic to speedup
            update_hctx_index = self.cache_update_index[
                layer_idx
            ]  # shape (batch_size, num_heads)
            batch_index = torch.arange(
                start=0, end=batch_size, device=device, dtype=torch.int64
            ).unsqueeze(
                1
            )  # to index shape (batch_size, num_heads, ...)

            self.mask_cache[layer_idx][batch_index, update_hctx_index] = 1
            if self.update_cache_content:
                self.value_cache[layer_idx][batch_index, update_hctx_index] = value_states[:, :, -1, :]
                self.key_cache[layer_idx][batch_index, update_hctx_index] = key_states[:, :, -1, :]

            # update valid cache length
            self.cache_valid_length[layer_idx] += (
                self.cache_valid_length[layer_idx] < self.cache_max_length[layer_idx]
            )

            is_circular = (
                self._kv_len[layer_idx] + 1 > self.static_cache_size[layer_idx]
            )[
                None, :
            ]  # shape (1, num_heads)
            self.cache_update_index[layer_idx] = (
                (
                    self.cache_update_index[layer_idx]
                    + 1
                    - self.circular_cache_head_index[layer_idx][None, :]
                )
                % self.circular_cache_size[layer_idx][None, :]
                + self.circular_cache_head_index[layer_idx][None, :]
            ) * is_circular + (self.cache_update_index[layer_idx] + 1) * (~is_circular)

            self._kv_len[layer_idx] += 1

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        else:
            # Use the attention mask to find the starting point of the valid inputs
            if attention_mask is not None:
                assert (
                    len(attention_mask.shape) == 2
                ), "The shape of the attention mask should be [batch_size, kv_len] and the dtype should be torch.int64."
                # attention_mask shape is (batch_size, self._kv_len[layer_idx])
                # take the newly added part of the attention mask
                attention_mask = attention_mask[
                    :, -seq_len:
                ]  # shape: [batch_size, seq_len]
            else:
                attention_mask = torch.ones(
                    (batch_size, seq_len), dtype=torch.int64, device=device
                )  # shape: [batch_size, seq_len]
            advancing_pos = (
                torch.cumsum(attention_mask, dim=-1, dtype=torch.int64) - 1
            )  # shape: [batch_size, seq_len]; For masked positions, they do not store in the cache nor advance the pos
            advancing_pos = torch.cat(
                [advancing_pos, advancing_pos[:, -1, None] + 1], dim=-1
            )  # shape: [batch_size, seq_len+1]
            reversed_advancing_pos = torch.flip(
                torch.cumsum(
                    torch.flip(attention_mask, dims=[-1]), dim=-1, dtype=torch.int64
                ),
                dims=[-1],
            )  # shape: [batch_size, seq_len]; For masked positions, they do not store in the cache nor advance the pos

            update_index_circular = (
                (
                    self.cache_update_index[layer_idx][:, None, :]
                    + advancing_pos[:, :, None]
                    - self.circular_cache_head_index[layer_idx][None, None, :]
                )
                % self.circular_cache_size[layer_idx][None, None, :]
            ) + self.circular_cache_head_index[layer_idx][
                None, None, :
            ]  # shape: (batch_size, seq_len+1, num_heads), ending at _kv_len > static_len
            update_index_static = (
                self.cache_update_index[layer_idx][:, None, :]
                + advancing_pos[:, :, None]
            )  # shape: (batch_size, seq_len+1, num_heads), ending at _kv_len <= static_len
            update_hctx_index = torch.where(
                self._kv_len[layer_idx] + advancing_pos[:, :, None]
                > self.static_cache_size[layer_idx][None, None, :],
                update_index_circular,
                update_index_static,
            )  # shape: (batch_size, seq_len+1, num_heads)

            # record index
            self.cache_update_index[layer_idx] = update_hctx_index[:, -1, :]
            update_hctx_index = update_hctx_index[:, :-1, :].permute(
                0, 2, 1
            )  # shape: (batch_size, num_heads, seq_len), value are in the realm of num_heads * seq_len
            batch_index = (
                torch.arange(batch_size, device=device)[:, None, None]
                .expand(-1, num_head, seq_len)
                .contiguous()
            )  # shape: (batch_size, num_heads, seq_len)

            # If the seq_len is so long that it > cache size, only the latter indexed key/value should be reserved
            # keep the index where ((A) the final size is less or equal than the circular cache size OR (B) index within static cache) AND (C) not masked
            valid_index_map = (
                (
                    reversed_advancing_pos[:, None, :]
                    <= self.circular_cache_size[layer_idx][None, :, None]
                )
                | (
                    update_hctx_index
                    < self.circular_cache_head_index[layer_idx][None, :, None]
                )
            ) & (
                attention_mask[:, None, :] == 1
            )  # shape (batch_size, num_heads, seq_len)

            # update valid cache length based on the valid index map
            self.cache_valid_length[layer_idx] += torch.sum(
                valid_index_map, dim=-1, dtype=torch.int64
            )
            self._kv_len[layer_idx] += seq_len

            # update cache
            valid_update_index = update_hctx_index[
                valid_index_map
            ].contiguous()  # shape: (batch_size * num_heads * seq_len)
            valid_batch_index = batch_index[
                valid_index_map
            ].contiguous()  # shape: (batch_size * num_heads * seq_len)

            self.mask_cache[layer_idx][valid_batch_index, valid_update_index] = (
                attention_mask[:, None, :].expand(-1, num_head, -1)[valid_index_map]
            )  # assign tensor of shape (batch_size * num_heads * seq_len)
            self.key_cache[layer_idx][valid_batch_index, valid_update_index, :] = (
                key_states[valid_index_map, :]
            )  # assign tensor of shape (batch_size * num_heads * seq_len, head_dim)
            self.value_cache[layer_idx][valid_batch_index, valid_update_index, :] = (
                value_states[valid_index_map, :]
            )  # assign tensor of shape (batch_size * num_heads * seq_len, head_dim)

            assert self.key_cache[layer_idx].is_contiguous(), "key_cache is not contiguous"
            assert self.value_cache[layer_idx].is_contiguous(), "value_cache is not contiguous"
            assert self.mask_cache[layer_idx].is_contiguous(), "mask_cache is not contiguous"

            # TODO: for multi-round conversation, how to deal with the caches from the previous rounds whose lengths are different. Ignoring all history for now
            return key_states, value_states

    def __len__(self):
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # ! warning: this function is meaningless, just a place holder
        return self._kv_len[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. This Cache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Reordering the cache is not implemented currently!")


def moa_config_to_cache_config(
    moa_config,
    seq_len,
    max_new_token: int = 1024,
    sink_size: int = 64,
    minimum_cache_size: int = 128,
    split_size: bool = 64,
    verbose: bool = True,
):
    """
    Convert the MoA configuration to the cache configuration

    Parameters:
        moa_config (`Dict`):
            The MoA configuration.
        seq_len (int):
            The sequence length.
        max_new_token (int, optional):
            The maximum number of new tokens. The N used to calculate cache size for each head equals seq_len + max_new_token. Defaults to 1024.
        sink_size (int, optional):
            The sink size. Defaults to 64.
        minimum_cache_size (int, optional):
            The minimum cache size. Defaults to 128.
        split_size (int, optional):
            The cache size of each head should be a multiple of this number. Defaults to 64.
        verbose (bool, optional):
            Whether to print the cache configuration summary. Defaults to True.

    Returns:
        A dictionary containing the cache configuration.
    """
    cache_size_dict = []
    static_size_dict = []

    alphas = moa_config["alphas"]
    betas = moa_config["betas"]

    for layer_id in range(len(alphas)):
        cache_size_this_layer = []
        static_size_this_layer = []
        for head_id in range(len(alphas[layer_id])):
            cache_size_this_head = int(
                alphas[layer_id][head_id]
                + (seq_len + max_new_token) * betas[layer_id][head_id]
            )
            cache_size_this_head = min(
                math.ceil(max(cache_size_this_head, minimum_cache_size) / split_size)
                * split_size,
                seq_len + max_new_token,
            )
            cache_size_this_layer.append(cache_size_this_head)
            static_size_this_layer.append(min(cache_size_this_head, sink_size))
        cache_size_dict.append(cache_size_this_layer)
        static_size_dict.append(static_size_this_layer)

    if verbose:
        print("Cache configuration")
        summary = []
        for layer_id in range(len(alphas)):
            for head_id in range(len(alphas[layer_id])):
                summary.append(
                    {
                        "layer_id": layer_id,
                        "head_id": head_id,
                        "raw_cache_size": seq_len + max_new_token,
                        "cache_size": cache_size_dict[layer_id][head_id],
                        "static_size": static_size_dict[layer_id][head_id],
                        "circular_size": cache_size_dict[layer_id][head_id]
                        - static_size_dict[layer_id][head_id],
                        "ratio": cache_size_dict[layer_id][head_id] / (seq_len + max_new_token),
                    }
                )
        summary = pd.DataFrame(summary)
        pd.options.display.float_format = (
            "{:.2f}".format
        )  # keep two digits for all values during printing
        # reduce the summary for each layer
        layer_summary = (
            summary.groupby("layer_id")
            .agg(
                {
                    "raw_cache_size": "mean",
                    "cache_size": ["mean", "min", "max"],
                    "static_size": ["mean", "min", "max"],
                    "circular_size": ["mean", "min", "max"],
                    "ratio": ["mean", "min", "max"],
                }
            )
            .reset_index()
        )
        print(layer_summary)
        # reduce the summary for the whole model
        model_summary = summary.agg(
            {
                "raw_cache_size": ["mean", "min", "max"],
                "cache_size": ["mean", "min", "max"],
                "static_size": ["mean", "min", "max"],
                "circular_size": ["mean", "min", "max"],
                "ratio": ["mean", "min", "max"],
            }
        ).T
        model_summary.columns = ["mean", "min", "max"]
        model_summary = model_summary.unstack().reset_index()
        model_summary.columns = ["metric", "stat", "value"]
        model_summary = model_summary.pivot(
            index="metric", columns="stat", values="value"
        ).reset_index()
        print(model_summary)

    return {
        "cache_size": cache_size_dict,
        "static_size": static_size_dict,
    }


if __name__ == "__main__":
    batch_size = 3
    head_dim = 4
    num_head = 2
    device = "cuda"

    cache = StaticCircularCache(
        cache_size=[[5, 10], [5, 5]],
        static_size=[[2, 3], [2, 3]],
        batch_size=batch_size,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
    )

    acc = 0
    seq_lens = [14, 9, 5]
    seq_len = max(seq_lens)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
    for i, this_seq_len in enumerate(seq_lens):
        attention_mask[i, -this_seq_len:] = 1

    for step in range(0, 10):
        key_states = (
            torch.arange(
                acc, num_head * seq_len + acc, dtype=torch.float16, device=device
            )[None, :, None]
            .expand(batch_size, -1, head_dim)
            .reshape(batch_size, num_head, seq_len, head_dim)
        )
        value_states = (
            torch.arange(
                acc, num_head * seq_len + acc, dtype=torch.float16, device=device
            )[None, :, None]
            .expand(batch_size, -1, head_dim)
            .reshape(batch_size, num_head, seq_len, head_dim)
        )

        acc += num_head * seq_len

        print(key_states)
        print(attention_mask)

        key_cache_new, value_cache_new = cache.update(
            key_states, value_states, 0, attention_mask
        )

        # if seq_len == 1:
        # list_key_cache_new = StaticCircularCache.to_uncontigious(key_cache_new, cache.head_index[0])
        group_key_cache_new = StaticCircularCache.to_group_contigious(
            cache.key_cache[0], cache.head_index[0]
        )
        group_attention_mask = StaticCircularCache.to_group_contigious(
            cache.mask_cache[0], cache.head_index[0]
        )
        # print(list_key_cache_new[0][1]) # head 0, total size 5, static size 2
        # print(list_key_cache_new[1][1]) # head 1, total size 10, static size 3
        for group in range(2):
            print(group_key_cache_new[group])
            print(group_attention_mask[group])

        print("done")

        seq_len = 1
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, seq_len, dtype=torch.int64, device=device),
            ],
            dim=-1,
        )