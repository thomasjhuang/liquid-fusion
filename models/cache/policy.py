from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheBudget:
    sink: int
    heavy: int
    recent: int

    def __post_init__(self):
        if self.sink < 0 or self.heavy < 0 or self.recent < 1:
            raise ValueError("sink and heavy budgets must be non-negative and recent must be positive")

    @property
    def capacity(self) -> int:
        return self.sink + self.heavy + self.recent


class SinkHeavyRecentPolicy:
    def __init__(self, budget: CacheBudget):
        self.budget = budget

    def select(self, scores: torch.Tensor) -> torch.LongTensor:
        batch_size, sequence_length = scores.shape
        if sequence_length <= self.budget.capacity:
            return torch.arange(sequence_length, device=scores.device).expand(batch_size, -1)

        sink_count = min(self.budget.sink, sequence_length)
        recent_count = min(self.budget.recent, sequence_length - sink_count)
        heavy_start = sink_count
        heavy_end = sequence_length - recent_count
        heavy_count = min(self.budget.heavy, heavy_end - heavy_start)

        parts = []
        if sink_count:
            parts.append(torch.arange(sink_count, device=scores.device).expand(batch_size, -1))
        if heavy_count:
            heavy_scores = scores[:, heavy_start:heavy_end]
            heavy_indices = torch.topk(
                heavy_scores,
                k=heavy_count,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            parts.append(heavy_indices + heavy_start)
        if recent_count:
            parts.append(
                torch.arange(
                    sequence_length - recent_count,
                    sequence_length,
                    device=scores.device,
                ).expand(batch_size, -1)
            )

        return torch.cat(parts, dim=-1).sort(dim=-1).values
