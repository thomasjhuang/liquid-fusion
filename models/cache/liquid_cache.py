from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers.cache_utils import Cache

from models.cache.policy import CacheBudget, SinkHeavyRecentPolicy


class LiquidFusionCache(Cache):
    def __init__(self, sink_size: int, heavy_budget: int, recent_budget: int):
        self.policy = SinkHeavyRecentPolicy(
            CacheBudget(sink=sink_size, heavy=heavy_budget, recent=recent_budget)
        )
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        self.score_cache: List[Tensor] = []
        self.position_cache: List[Tensor] = []
        self.seen_tokens = 0

    @property
    def capacity(self) -> int:
        return self.policy.budget.capacity

    def __getitem__(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        if layer_idx >= len(self):
            raise KeyError(f"Cache has {len(self)} layers, requested layer {layer_idx}")
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self[layer_idx]

    def __len__(self) -> int:
        return len(self.key_cache)

    def _normalize_positions(
        self,
        position_ids: Optional[Tensor],
        key_states: Tensor,
    ) -> Tensor:
        batch_size, _, sequence_length, _ = key_states.shape
        if position_ids is None:
            start = self.seen_tokens
            position_ids = torch.arange(
                start,
                start + sequence_length,
                device=key_states.device,
            ).unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)
        expected_shape = (batch_size, sequence_length)
        if position_ids.shape != expected_shape:
            raise ValueError(
                f"position_ids must have shape {expected_shape}, got {tuple(position_ids.shape)}"
            )
        return position_ids.to(device=key_states.device, dtype=torch.long)

    def _gather(self, tensor: Tensor, indices: Tensor) -> Tensor:
        expanded = indices[:, None, :, None].expand(
            tensor.shape[0],
            tensor.shape[1],
            indices.shape[1],
            tensor.shape[3],
        )
        return torch.gather(tensor, dim=2, index=expanded)

    def _select(self, layer_idx: int) -> None:
        indices = self.policy.select(self.score_cache[layer_idx])
        self.key_cache[layer_idx] = self._gather(self.key_cache[layer_idx], indices)
        self.value_cache[layer_idx] = self._gather(self.value_cache[layer_idx], indices)
        self.score_cache[layer_idx] = torch.gather(
            self.score_cache[layer_idx],
            dim=1,
            index=indices,
        )
        self.position_cache[layer_idx] = torch.gather(
            self.position_cache[layer_idx],
            dim=1,
            index=indices,
        )

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        cache_kwargs = cache_kwargs or {}
        positions = self._normalize_positions(cache_kwargs.get("position_ids"), key_states)
        sequence_length = key_states.shape[-2]

        if layer_idx < len(self.key_cache) and sequence_length != 1:
            raise ValueError("Chunked decoding is not supported")

        if layer_idx == 0:
            self.seen_tokens += sequence_length

        if layer_idx >= len(self.key_cache):
            if layer_idx != len(self.key_cache):
                raise ValueError(f"Layers must update in order, received layer {layer_idx}")
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.score_cache.append(
                torch.zeros(
                    key_states.shape[0],
                    sequence_length,
                    dtype=torch.float32,
                    device=key_states.device,
                )
            )
            self.position_cache.append(positions)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states],
                dim=-2,
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states],
                dim=-2,
            )
            self.score_cache[layer_idx] = torch.cat(
                [
                    self.score_cache[layer_idx],
                    torch.zeros(
                        key_states.shape[0],
                        sequence_length,
                        dtype=torch.float32,
                        device=key_states.device,
                    ),
                ],
                dim=-1,
            )
            self.position_cache[layer_idx] = torch.cat(
                [self.position_cache[layer_idx], positions],
                dim=-1,
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def record_attention(self, layer_idx: int, attention_probs: Tensor) -> None:
        attention_mass = attention_probs.detach().float().mean(dim=1).sum(dim=1)
        if attention_mass.shape != self.score_cache[layer_idx].shape:
            raise ValueError(
                "Attention scores and cache entries must have matching batch and sequence dimensions"
            )
        self.score_cache[layer_idx] = self.score_cache[layer_idx] + attention_mass
        if self.get_seq_length(layer_idx) > self.capacity:
            self._select(layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None or layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx in range(len(self)):
            key_device = self.key_cache[layer_idx].device
            score_device = self.score_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(key_device)
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(key_device)
            )
            self.score_cache[layer_idx] = self.score_cache[layer_idx].index_select(
                0, beam_idx.to(score_device)
            )
            self.position_cache[layer_idx] = self.position_cache[layer_idx].index_select(
                0, beam_idx.to(score_device)
            )

    def to_legacy_cache(self) -> Tuple[Tuple[Tensor, Tensor], ...]:
        return tuple(self[layer_idx] for layer_idx in range(len(self)))
