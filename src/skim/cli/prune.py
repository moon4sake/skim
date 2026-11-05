from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skim.config import ALPHA, FAMILY_RATIO, LAYER_SCHEDULE, MAX_SPARSITY_DRIFT, MODEL_REGISTRY
from skim.io import load_model, load_tokenizer, save_json
from skim.prune.allocation import SCHEDULES, prunable_param_total
from skim.prune.masks import MaskApplier, build_weight_masks
from skim.prune.pruner import (
    prune,
    realised_family_rates,
    realised_prunable_sparsity,
    removed_params,
)
from skim.prune.units import family_param_totals
from skim.records import add_stats, iter_record_shards


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--alpha_c", type=float, default=ALPHA[0])
    parser.add_argument("--alpha_d", type=float, default=ALPHA[1])
    parser.add_argument("--alpha_f", type=float, default=ALPHA[2])
    parser.add_argument("--layer_schedule", choices=list(SCHEDULES), default=LAYER_SCHEDULE)
    return parser.parse_args(argv)


def prune_and_save(
    model, record_batches, sparsity: float, alpha, schedule: str, output_dir, info: dict
) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_before = sum(p.numel() for p in model.parameters())
    plan, rates = prune(model, record_batches, sparsity, alpha=alpha, schedule=schedule)
    masks = build_weight_masks(model, plan)
    MaskApplier.apply(model, masks)
    MaskApplier.assert_applied(model, masks)
    P, removed = family_param_totals(model), removed_params(model, plan)
    realised = realised_prunable_sparsity(model, plan)
    print(f"[prune] prunable-param sparsity: requested {sparsity:.4f}, realised {realised:.4f}")
    if abs(realised - sparsity) > MAX_SPARSITY_DRIFT:
        raise RuntimeError(f"realised sparsity {realised:.4f} drifts from requested {sparsity:.4f}")
    total_after = total_before - int(sum(removed.values()))
    artifact = {
        "method": "skim",
        **info,
        "target_sparsity": float(sparsity),
        "alpha": list(alpha),
        "layer_schedule": schedule,
        "family_ratio": dict(FAMILY_RATIO),
        "family_rates": rates,
        "realised_family_rates": realised_family_rates(model, plan),
        "family_param_totals": P,
        "prunable_param_total": prunable_param_total(P),
        "prunable_param_sparsity": realised,
        "units_removed": plan.counts(),
        "total_params_before": total_before,
        "total_params_after": total_after,
        "whole_model_sparsity": 1.0 - total_after / total_before,
        "structured_plan": plan.to_artifact(),
    }
    torch.save({k: m.bool().cpu() for k, m in masks.items()}, output_dir / "pruning_masks.pt")
    save_json(output_dir / "pruning_artifact.json", artifact)
    return artifact


def run_prune(args: argparse.Namespace) -> dict:
    model_name = MODEL_REGISTRY[args.model]
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, args.device)
    calibration: dict = {}

    def record_batches():
        for records, stats in iter_record_shards(tokenizer):
            add_stats(calibration, stats)
            yield records

    info = {"model": args.model, "model_name": model_name, "calibration_records": calibration}
    alpha = (args.alpha_c, args.alpha_d, args.alpha_f)
    return prune_and_save(
        model, record_batches(), args.sparsity, alpha, args.layer_schedule, args.output_dir, info
    )


def main(argv=None) -> None:
    run_prune(parse_args(argv))


if __name__ == "__main__":
    main()
