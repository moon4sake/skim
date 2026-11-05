from __future__ import annotations

import torch

from skim.config import ALPHA, LAMBDA_HAT_H, LAMBDA_KD, OBJECTIVE, TEMPERATURE

OBJECTIVES = ("skim", "vanilla")
CHUNK = 128


def _forward_kl(student_logits, teacher_logits, tau, support=None):
    n = student_logits.shape[0]
    total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, n, CHUNK):
        s = student_logits[start : start + CHUNK].float() / tau
        t = teacher_logits[start : start + CHUNK].float() / tau
        if support is None:
            s_lse, t_lse = s.logsumexp(-1, keepdim=True), t.logsumexp(-1, keepdim=True)
            p = (t - t_lse).exp()
        else:
            outside = ~support[start : start + CHUNK]
            s_lse = s.masked_fill(outside, -float("inf")).logsumexp(-1, keepdim=True)
            t_lse = t.masked_fill(outside, -float("inf")).logsumexp(-1, keepdim=True)
            p = (t - t_lse).exp().masked_fill(outside, 0.0)
        total = total + (p * ((t - t_lse) - (s - s_lse))).sum()
    return tau**2 * total / max(1, n)


def constrained_kl(student_logits, teacher_logits, valid, tau: float = TEMPERATURE):
    return _forward_kl(student_logits, teacher_logits, tau, valid)


def vanilla_kl(student_logits, teacher_logits, tau: float = TEMPERATURE):
    return _forward_kl(student_logits, teacher_logits, tau)


def _matched_layers(num_blocks: int) -> list[int]:
    l0 = (num_blocks - 4) // 2
    if l0 < 1:
        raise ValueError(f"hidden-state matching needs at least 6 blocks, got {num_blocks}")
    return [l0, l0 + 2, l0 + 4]


def hidden_loss(student_hidden, teacher_hidden, context, decision, format, alpha=ALPHA):
    device = student_hidden[0].device
    alpha_hat = torch.tensor(alpha, dtype=torch.float32, device=device) / sum(alpha)
    groups = (context, decision, format)
    index = torch.tensor([p for g in groups for p in g], dtype=torch.long, device=device)
    weight = torch.cat([alpha_hat[c].expand(len(g)) for c, g in enumerate(groups)])
    if index.numel() == 0 or float(weight.sum()) == 0.0:
        return torch.zeros((), device=device, dtype=torch.float32)
    layers = _matched_layers(len(student_hidden) - 1)
    total = torch.zeros((), device=device, dtype=torch.float32)
    for layer in layers:
        diff = student_hidden[layer].index_select(0, index).float() - teacher_hidden[
            layer
        ].index_select(0, index.to(teacher_hidden[layer].device)).float().to(device)
        total = total + (weight * diff.pow(2).sum(-1)).sum()
    return total / (len(layers) * weight.sum())


def skim_loss(
    l_kd,
    l_hidden,
    l_vanilla=None,
    *,
    lambda_hat_h: float = LAMBDA_HAT_H,
    lambda_kd: float = LAMBDA_KD,
    objective: str = OBJECTIVE,
):
    if objective == "vanilla":
        return l_vanilla
    total = l_kd + lambda_hat_h * l_hidden
    if lambda_kd:
        total = total + lambda_kd * l_vanilla
    return total
