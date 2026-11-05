from __future__ import annotations

import torch

CATEGORIES = ("context", "decision", "format")


def check_alpha(alpha) -> tuple[float, float, float]:
    alpha = tuple(float(a) for a in alpha)
    if len(alpha) != 3 or min(alpha) < 0.0 or sum(alpha) <= 0.0:
        raise ValueError(f"alpha must be three non-negative weights (c, d, f), got {alpha}")
    return alpha


def category_mean_square(sum_sq: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return torch.zeros_like(sum_sq, dtype=torch.float32)
    return sum_sq.float() / float(count)


def surrogate_precision(sums, alpha) -> torch.Tensor:
    m_c, m_d, m_f = (category_mean_square(sums.sum_sq[k], sums.count[k]) for k in CATEGORIES)
    a_c, a_d, a_f = alpha
    return (a_c * m_c + a_d * m_d + a_f * m_f).cpu()


def column_norm_sq(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().float().pow(2).sum(dim=0).cpu()


def projected_head_output_energy(a: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    num_heads, head_dim = a.shape[1], a.shape[2]
    w = o_proj_weight.detach().float().view(-1, num_heads, head_dim)
    return torch.einsum("thd,ohd->tho", a.float(), w).pow(2).sum(dim=-1)


def mlp_neuron_saliency(sums, down_proj_weight: torch.Tensor, alpha) -> torch.Tensor:
    return surrogate_precision(sums, alpha) * column_norm_sq(down_proj_weight)


def linear_attention_group_saliency(
    sums, out_proj_weight: torch.Tensor, group_width: int, alpha
) -> torch.Tensor:
    channels = surrogate_precision(sums, alpha) * column_norm_sq(out_proj_weight)
    return channels.view(-1, group_width).sum(dim=1)


def kv_group_saliency(sums, kv_group_size: int, alpha) -> torch.Tensor:
    return surrogate_precision(sums, alpha).view(-1, kv_group_size).sum(dim=1)


def score_units(accumulator, alpha) -> dict[str, dict[int, torch.Tensor]]:
    scores: dict[str, dict[int, torch.Tensor]] = {}
    for family, layers in accumulator.stats.items():
        scores[family] = {}
        for layer_idx, (sums, module, width) in layers.items():
            if family == "mlp":
                score = mlp_neuron_saliency(sums, module.weight, alpha)
            elif family == "linear_attn":
                score = linear_attention_group_saliency(sums, module.weight, width, alpha)
            else:
                score = kv_group_saliency(sums, width, alpha)
            scores[family][layer_idx] = score
    return scores
