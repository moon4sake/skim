from __future__ import annotations

from collections.abc import Iterable

from skim.config import ALPHA, FAMILY_RATIO, LAYER_SCHEDULE
from skim.prune.allocation import SCHEDULES, family_rates, prunable_param_total
from skim.prune.calibration import SaliencyAccumulator
from skim.prune.masks import FAMILIES, MaskPlan
from skim.prune.saliency import check_alpha, score_units
from skim.prune.units import (
    family_param_totals,
    kv_group_param_count,
    linear_attention_group_param_count,
    mlp_neuron_param_count,
)


def prune(
    model,
    record_batches: Iterable[list],
    sparsity: float,
    *,
    alpha=ALPHA,
    schedule: str = LAYER_SCHEDULE,
) -> tuple[MaskPlan, dict[str, float]]:
    alpha = check_alpha(alpha)
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown layer schedule {schedule!r}; choose from {list(SCHEDULES)}")
    rates = family_rates(family_param_totals(model), sparsity)
    accumulator = SaliencyAccumulator(model, [f for f in FAMILIES if FAMILY_RATIO[f] > 0.0])
    try:
        n_records = sum(accumulator.accumulate(records) for records in record_batches)
    finally:
        accumulator.close()
    if n_records == 0:
        raise RuntimeError("pruning needs at least one calibration record")
    scores = score_units(accumulator, alpha)
    allocate = SCHEDULES[schedule]
    plan = MaskPlan(**{f: allocate(scores[f], rates[f]) if f in scores else {} for f in FAMILIES})
    return plan, rates


def removed_params(model, plan: MaskPlan) -> dict[str, float]:
    per_unit = {
        "mlp": lambda layer: mlp_neuron_param_count(model),
        "linear_attn": lambda layer: linear_attention_group_param_count(model, layer),
        "attention": lambda layer: kv_group_param_count(model, layer),
    }
    return {
        f: float(sum(per_unit[f](layer) * len(units) for layer, units in getattr(plan, f).items()))
        for f in FAMILIES
    }


def realised_prunable_sparsity(model, plan: MaskPlan) -> float:
    P = family_param_totals(model)
    return sum(removed_params(model, plan).values()) / prunable_param_total(P)


def realised_family_rates(model, plan: MaskPlan) -> dict[str, float]:
    P, removed = family_param_totals(model), removed_params(model, plan)
    return {f: removed[f] / P[f] if P[f] > 0 else 0.0 for f in FAMILIES}
