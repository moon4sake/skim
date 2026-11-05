from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from skim.labels import Record, predicting_state
from skim.prune.saliency import CATEGORIES, projected_head_output_energy
from skim.prune.units import kv_group_geometry, linear_attention_group_geometry


@dataclass
class CategorySums:
    sum_sq: dict[str, torch.Tensor]
    count: dict[str, int]

    @classmethod
    def zeros(cls, n: int, device) -> CategorySums:
        sums = {k: torch.zeros(n, dtype=torch.float32, device=device) for k in CATEGORIES}
        return cls(sums, {k: 0 for k in CATEGORIES})

    def add(self, x: torch.Tensor, positions: dict[str, torch.Tensor], energy) -> None:
        for k, pos in positions.items():
            if pos.numel():
                self.sum_sq[k] += energy(x.index_select(0, pos)).sum(dim=0)
                self.count[k] += int(pos.numel())


def category_positions(record: Record, device) -> dict[str, torch.Tensor]:
    positions = {
        "context": record.context,
        "decision": predicting_state(record.decision),
        "format": predicting_state(record.format),
    }
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in positions.items()}


def square(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.float().pow(2)


class SaliencyAccumulator:
    def __init__(self, model, families):
        self.model = model
        self.device = next(model.parameters()).device
        self.positions: dict[str, torch.Tensor] | None = None
        self.stats: dict[str, dict] = {family: {} for family in families}
        self.hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            if "mlp" in families:
                down_proj = layer.mlp.down_proj
                self._watch("mlp", layer_idx, down_proj, down_proj.in_features, None, square)
            if "linear_attn" in families and hasattr(layer, "linear_attn"):
                out_proj = layer.linear_attn.out_proj
                width = linear_attention_group_geometry(model, layer_idx)["out_cols_per_group"]
                self._watch("linear_attn", layer_idx, out_proj, out_proj.in_features, width, square)
            if "attention" in families and hasattr(layer, "self_attn"):
                geometry = kv_group_geometry(model, layer_idx)
                o_proj, num_heads = layer.self_attn.o_proj, geometry["num_heads"]

                def head_energy(tokens, o_proj=o_proj, num_heads=num_heads):
                    a = tokens.view(tokens.shape[0], num_heads, -1)
                    return projected_head_output_energy(a, o_proj.weight)

                group_size = geometry["kv_group_size"]
                self._watch("attention", layer_idx, o_proj, num_heads, group_size, head_energy)

    def _watch(self, family: str, layer_idx: int, module, n: int, width, energy) -> None:
        sums = CategorySums.zeros(n, module.weight.device)
        self.stats[family][layer_idx] = (sums, module, width)

        def hook(_module, inputs):
            if self.positions is not None:
                sums.add(inputs[0][0], self.positions, energy)

        self.hooks.append(module.register_forward_pre_hook(hook))

    @torch.no_grad()
    def accumulate(self, records: list[Record]) -> int:
        self.model.eval()
        for record in tqdm(records, desc="calibration"):
            input_ids = torch.tensor([record.input_ids], dtype=torch.long, device=self.device)
            self.positions = category_positions(record, self.device)
            self.model.model(input_ids=input_ids, use_cache=False)
        self.positions = None
        return len(records)

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
