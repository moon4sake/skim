from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from skim.config import (
    ALPHA,
    BATCH,
    EPOCHS,
    GRAD_ACCUM,
    GRAD_CLIP,
    KD_SCHEDULE,
    LAMBDA_HAT_H,
    LAMBDA_KD,
    LR,
    LR_WARMUP_FRAC,
    MAX_SEQ_LEN,
    OBJECTIVE,
    OPD_WARMUP_FRAC,
    ROLLOUT_MAX_NEW_TOKENS,
    SEED,
    TEMPERATURE,
    WEIGHT_DECAY,
)
from skim.kd.data import collate, epoch_batches, optimizer_steps, output_rows
from skim.kd.losses import (
    OBJECTIVES,
    constrained_kl,
    hidden_loss,
    skim_loss,
    vanilla_kl,
)
from skim.kd.opd import SCHEDULES, on_policy_batch, phase, rollout_counts
from skim.labels import Record, predicting_state
from skim.prune.masks import MaskApplier
from skim.prune.saliency import check_alpha


@dataclass(frozen=True)
class KDConfig:
    alpha: tuple[float, float, float] = ALPHA
    epochs: int = EPOCHS
    lr: float = LR
    batch: int = BATCH
    grad_accum: int = GRAD_ACCUM
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    lr_warmup_frac: float = LR_WARMUP_FRAC
    seed: int = SEED
    temperature: float = TEMPERATURE
    lambda_hat_h: float = LAMBDA_HAT_H
    lambda_kd: float = LAMBDA_KD
    objective: str = OBJECTIVE
    kd_schedule: str = KD_SCHEDULE
    opd_warmup_frac: float = OPD_WARMUP_FRAC
    rollout_max_new_tokens: int = ROLLOUT_MAX_NEW_TOKENS
    max_seq_len: int = MAX_SEQ_LEN

    def __post_init__(self):
        object.__setattr__(self, "alpha", check_alpha(self.alpha))
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.kd_schedule not in SCHEDULES:
            raise ValueError(f"kd_schedule must be one of {SCHEDULES}, got {self.kd_schedule!r}")
        if self.objective == "vanilla":
            object.__setattr__(self, "kd_schedule", "off_policy")
        if self.lambda_hat_h < 0 or self.lambda_kd < 0:
            raise ValueError("lambda_hat_h and lambda_kd must be non-negative")

    @property
    def uses_vanilla(self) -> bool:
        return self.objective == "vanilla" or self.lambda_kd > 0


def pruned_weights(
    model, masks: dict[str, torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    MaskApplier.validate(model, masks)
    pairs = []
    for key, mask in masks.items():
        weight = model.get_submodule(key.removesuffix(".weight")).weight
        pairs.append((weight, mask.to(weight.device) == 0))
    return pairs


def apply_masks(pairs) -> None:
    with torch.no_grad():
        for weight, pruned in pairs:
            weight.masked_fill_(pruned, 0.0)


def optimizer_step(optimizer, params, pairs, grad_clip: float, counts: dict) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        counts["nonfinite_grad"] += 1
    optimizer.zero_grad(set_to_none=True)
    apply_masks(pairs)
    return float(grad_norm)


def lr_factor(step: int, total_steps: int, warmup_frac: float = LR_WARMUP_FRAC) -> float:
    warmup = math.ceil(total_steps * warmup_frac)
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))


def batch_losses(student, teacher, batch: list[Record], cfg: KDConfig, pad_id: int) -> dict:
    device = next(student.parameters()).device
    t_device = next(teacher.parameters()).device
    input_ids, attention_mask = collate(batch, pad_id, device)
    keep, rows = output_rows(batch, input_ids.shape[1])
    skim = cfg.objective == "skim"
    kwargs = {"use_cache": False, "output_hidden_states": skim, "logits_to_keep": keep}
    with torch.no_grad():
        t_out = teacher(
            input_ids=input_ids.to(t_device),
            attention_mask=attention_mask.to(t_device),
            **kwargs,
        )
    s_out = student(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    terms: dict[str, list] = {"constrained_kl": [], "hidden": [], "vanilla_kl": []}
    for b, record in enumerate(batch):
        s_logits = s_out.logits[b, rows[b]]
        t_logits = t_out.logits[b, rows[b]].to(device)
        if skim:
            valid = record.valid_sets.mask(width=s_logits.shape[-1], device=device)
            terms["constrained_kl"].append(
                constrained_kl(s_logits, t_logits, valid, cfg.temperature)
            )
            terms["hidden"].append(
                hidden_loss(
                    tuple(h[b] for h in s_out.hidden_states),
                    tuple(h[b] for h in t_out.hidden_states),
                    record.context,
                    predicting_state(record.decision),
                    predicting_state(record.format),
                    cfg.alpha,
                )
            )
        if cfg.uses_vanilla:
            terms["vanilla_kl"].append(vanilla_kl(s_logits, t_logits, cfg.temperature))
    return {k: torch.stack(v).mean() if v else None for k, v in terms.items()}


def train(
    student,
    teacher,
    tokenizer,
    records: list[Record],
    masks: dict[str, torch.Tensor],
    cfg: KDConfig | None = None,
    log_path: str | Path | None = None,
) -> dict:
    cfg = cfg or KDConfig()
    device = next(student.parameters()).device
    params = [p for p in student.parameters() if p.requires_grad]
    teacher.eval().requires_grad_(False)
    student.train()
    pairs = pruned_weights(student, masks)
    apply_masks(pairs)

    total_steps = optimizer_steps(len(records), cfg.epochs, cfg.batch, cfg.grad_accum)
    n_micro = cfg.epochs * math.ceil(len(records) / cfg.batch)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, cfg.lr_warmup_frac)
    )
    counts = {"nonfinite_loss": 0, "nonfinite_grad": 0, **rollout_counts()}
    log = {
        "config": asdict(cfg),
        "records": len(records),
        "optimizer_steps": total_steps,
        "counts": counts,
        "steps": [],
    }
    pad_id = tokenizer.pad_token_id

    step, micro, window = 0, 0, []
    for epoch in range(cfg.epochs):
        for batch in epoch_batches(records, epoch, cfg.batch, cfg.seed):
            current = phase(step, total_steps, cfg.kd_schedule, cfg.opd_warmup_frac)
            if current == "on_policy":
                batch = on_policy_batch(
                    student,
                    batch,
                    tokenizer,
                    counts,
                    device,
                    cfg.rollout_max_new_tokens,
                    cfg.max_seq_len,
                )
            terms = batch_losses(student, teacher, batch, cfg, pad_id)
            loss = skim_loss(
                terms["constrained_kl"],
                terms["hidden"],
                terms["vanilla_kl"],
                lambda_hat_h=cfg.lambda_hat_h,
                lambda_kd=cfg.lambda_kd,
                objective=cfg.objective,
            )
            window_size = min(cfg.grad_accum, n_micro - step * cfg.grad_accum)
            if torch.isfinite(loss):
                (loss / window_size).backward()
                window.append({k: float(v) for k, v in terms.items() if v is not None})
                window[-1]["total"] = float(loss)
            else:
                counts["nonfinite_loss"] += 1
            micro += 1
            if micro % cfg.grad_accum and micro < n_micro:
                continue
            grad_norm = optimizer_step(optimizer, params, pairs, cfg.grad_clip, counts)
            means = {k: sum(w[k] for w in window) / len(window) for k in (window or [{}])[0]}
            lr = scheduler.get_last_lr()[0]
            entry = {"step": step, "phase": current, "lr": lr, "grad_norm": grad_norm, **means}
            log["steps"].append(entry)
            scheduler.step()
            step, window = step + 1, []
            if step == 1 or step % 25 == 0 or step == total_steps:
                print(f"[kd] {json.dumps(entry)}", flush=True)
        if log_path is not None:
            Path(log_path).write_text(json.dumps(log, indent=2))
    student.eval()
    MaskApplier.assert_applied(student, masks)
    log["completed"] = True
    if log_path is not None:
        Path(log_path).write_text(json.dumps(log, indent=2))
    return log
