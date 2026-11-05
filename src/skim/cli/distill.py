from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from skim import config
from skim.config import MODEL_REGISTRY
from skim.io import load_model, load_tokenizer, save_json
from skim.kd.lora import lora_student, merge_lora
from skim.kd.losses import OBJECTIVES
from skim.kd.opd import SCHEDULES
from skim.kd.trainer import KDConfig, train
from skim.prune.masks import MaskApplier
from skim.records import xlam_records


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher_device", default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH)
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--lambda_hat_H", type=float, default=config.LAMBDA_HAT_H)
    parser.add_argument("--lambda_KD", type=float, default=config.LAMBDA_KD)
    parser.add_argument("--objective", choices=OBJECTIVES, default=config.OBJECTIVE)
    parser.add_argument("--kd_schedule", choices=SCHEDULES, default=config.KD_SCHEDULE)
    args = parser.parse_args(argv)
    if args.lr is None:
        args.lr = config.LORA_LR if args.lora else config.LR
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_distill(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = Path(args.mask_path).expanduser().resolve()
    artifact = json.loads((mask_path.parent / "pruning_artifact.json").read_text())
    cfg = KDConfig(
        alpha=artifact["alpha"],
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        temperature=args.temperature,
        lambda_hat_h=args.lambda_hat_H,
        lambda_kd=args.lambda_KD,
        objective=args.objective,
        kd_schedule=args.kd_schedule,
    )
    model_name = MODEL_REGISTRY[args.model]
    tokenizer = load_tokenizer(model_name)
    masks = torch.load(mask_path, map_location="cpu")
    teacher = load_model(model_name, args.teacher_device or args.device, dtype=torch.bfloat16)
    if args.lora:
        student = lora_student(model_name, masks, args.device)
    else:
        student = load_model(model_name, args.device)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(
        f"[kd] student {'NF4 + LoRA' if args.lora else next(student.parameters()).dtype}, "
        f"{trainable:,} trainable parameters; teacher {next(teacher.parameters()).dtype}",
        flush=True,
    )
    records, record_stats = xlam_records(tokenizer)
    log = train(
        student,
        teacher,
        tokenizer,
        records,
        {} if args.lora else masks,
        cfg,
        output_dir / "training_log.json",
    )

    final_dir = output_dir / "final"
    if args.lora:
        student = merge_lora(student)
        MaskApplier.apply(student, masks)
    else:
        student.to(torch.bfloat16)
        MaskApplier.assert_applied(student, masks)
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    summary = {
        "method": cfg.objective,
        "model": args.model,
        "mask_path": str(mask_path),
        "model_path": str(final_dir),
        "saved_dtype": str(next(student.parameters()).dtype).removeprefix("torch."),
        "config": asdict(cfg),
        "records": record_stats,
        "optimizer_steps": log["optimizer_steps"],
        "counts": log["counts"],
        "args": vars(args),
    }
    save_json(output_dir / "kd_summary.json", summary)
    return summary


def main(argv=None) -> None:
    run_distill(parse_args(argv))


if __name__ == "__main__":
    main()
