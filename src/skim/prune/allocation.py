from __future__ import annotations

import math

import torch

from skim.config import FAMILY_RATIO


def check_rate(rate: float) -> float:
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"pruning rate must lie in [0, 1), got {rate}")
    return float(rate)


def prunable_param_total(P: dict[str, float], ratio: dict[str, float] = FAMILY_RATIO) -> float:
    return float(sum(P[f] for f in P if ratio[f] > 0.0))


def family_rates(
    P: dict[str, float], sparsity: float, ratio: dict[str, float] = FAMILY_RATIO
) -> dict[str, float]:
    check_rate(sparsity)
    scale = sparsity * prunable_param_total(P, ratio) / sum(ratio[f] * P[f] for f in P)
    return {f: check_rate(scale * ratio[f]) for f in P}


def standardize_within_layer(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.double()
    std = scores.std(correction=0)
    if std == 0:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / std


def adaptive_schedule(scores_by_layer: dict[int, torch.Tensor], rate: float) -> dict[int, set[int]]:
    layers = sorted(scores_by_layer)
    units = [(layer, i) for layer in layers for i in range(len(scores_by_layer[layer]))]
    z = torch.cat([standardize_within_layer(scores_by_layer[layer]) for layer in layers])
    budget = math.floor(check_rate(rate) * len(units))
    removed: dict[int, set[int]] = {layer: set() for layer in layers}
    for flat in torch.argsort(z, stable=True).tolist():
        if budget == 0:
            break
        layer, i = units[flat]
        if len(removed[layer]) < len(scores_by_layer[layer]) - 1:
            removed[layer].add(i)
            budget -= 1
    return {layer: idx for layer, idx in removed.items() if idx}


def uniform_schedule(scores_by_layer: dict[int, torch.Tensor], rate: float) -> dict[int, set[int]]:
    check_rate(rate)
    removed: dict[int, set[int]] = {}
    for layer, scores in sorted(scores_by_layer.items()):
        k = min(len(scores) - 1, math.floor(rate * len(scores) + 0.5))
        if k > 0:
            removed[layer] = set(torch.argsort(scores, stable=True)[:k].tolist())
    return removed


SCHEDULES = {"adaptive": adaptive_schedule, "uniform": uniform_schedule}
