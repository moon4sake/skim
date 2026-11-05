from __future__ import annotations

from dataclasses import dataclass

import torch

from skim.prune.units import kv_group_geometry, linear_attention_group_geometry

FAMILIES = ("mlp", "linear_attn", "attention")


@dataclass(frozen=True)
class MaskPlan:
    attention: dict[int, set[int]]
    linear_attn: dict[int, set[int]]
    mlp: dict[int, set[int]]

    def to_artifact(self) -> dict:
        return {
            name: {"remove_per_layer": {str(k): sorted(v) for k, v in sorted(units.items())}}
            for name, units in vars(self).items()
        }

    def counts(self) -> dict[str, int]:
        return {
            "attention_groups": sum(len(v) for v in self.attention.values()),
            "linear_attn_groups": sum(len(v) for v in self.linear_attn.values()),
            "mlp_neurons": sum(len(v) for v in self.mlp.values()),
        }


class MaskApplier:
    @staticmethod
    def _masked_modules(model, masks: dict[str, torch.Tensor]):
        modules = dict(model.named_modules())
        missing = [k for k in masks if k.removesuffix(".weight") not in modules]
        if missing:
            raise ValueError(f"Mask keys do not match model modules: {missing[:10]}")
        for key, mask in masks.items():
            yield key, modules[key.removesuffix(".weight")], mask

    @classmethod
    def validate(cls, model, masks: dict[str, torch.Tensor]) -> None:
        list(cls._masked_modules(model, masks))

    @classmethod
    def apply(cls, model, masks: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for _key, module, mask in cls._masked_modules(model, masks):
                module.weight.mul_(mask.to(module.weight.device, dtype=module.weight.dtype))

    @classmethod
    def assert_applied(cls, model, masks: dict[str, torch.Tensor]) -> None:
        bad = [
            key
            for key, module, mask in cls._masked_modules(model, masks)
            if bool((module.weight.detach()[mask.to(module.weight.device) == 0] != 0).any())
        ]
        if bad:
            raise ValueError(f"Masked weights are not zero: {bad[:10]}")


def mlp_neuron_weight_masks(
    prefix: str, remove_neurons: set[int], *, intermediate_size: int, hidden_size: int
) -> dict[str, torch.Tensor]:
    gate_mask = torch.ones(intermediate_size, hidden_size)
    up_mask = torch.ones(intermediate_size, hidden_size)
    down_mask = torch.ones(hidden_size, intermediate_size)
    for n in remove_neurons:
        gate_mask[n, :] = 0
        up_mask[n, :] = 0
        down_mask[:, n] = 0
    return {
        f"{prefix}.mlp.gate_proj": gate_mask,
        f"{prefix}.mlp.up_proj": up_mask,
        f"{prefix}.mlp.down_proj": down_mask,
    }


def block(group: int, width: int, offset: int = 0) -> slice:
    return slice(offset + group * width, offset + (group + 1) * width)


def kv_group_weight_masks(
    model, layer_idx: int, prefix: str, remove_groups: set[int]
) -> dict[str, torch.Tensor]:
    geometry = kv_group_geometry(model, layer_idx)
    attn = model.model.layers[layer_idx].self_attn
    q_mask = torch.ones_like(attn.q_proj.weight, dtype=torch.float32)
    o_mask = torch.ones_like(attn.o_proj.weight, dtype=torch.float32)
    k_mask = torch.ones_like(attn.k_proj.weight, dtype=torch.float32)
    v_mask = torch.ones_like(attn.v_proj.weight, dtype=torch.float32)
    q_rows = geometry["kv_group_size"] * geometry["q_rows_per_head"]
    o_cols = geometry["kv_group_size"] * geometry["o_cols_per_head"]
    kv_rows = geometry["kv_rows_per_head"]
    for g in remove_groups:
        q_mask[block(g, q_rows), :] = 0
        o_mask[:, block(g, o_cols)] = 0
        k_mask[block(g, kv_rows), :] = 0
        v_mask[block(g, kv_rows), :] = 0
    return {
        f"{prefix}.self_attn.q_proj": q_mask,
        f"{prefix}.self_attn.o_proj": o_mask,
        f"{prefix}.self_attn.k_proj": k_mask,
        f"{prefix}.self_attn.v_proj": v_mask,
    }


def linear_attention_group_weight_masks(
    model, layer_idx: int, prefix: str, remove_groups: set[int]
) -> dict[str, torch.Tensor]:
    geometry = linear_attention_group_geometry(model, layer_idx)
    attn = model.model.layers[layer_idx].linear_attn
    qkv_mask = torch.ones_like(attn.in_proj_qkv.weight, dtype=torch.float32)
    conv_mask = torch.ones_like(attn.conv1d.weight, dtype=torch.float32)
    z_mask = torch.ones_like(attn.in_proj_z.weight, dtype=torch.float32)
    a_mask = torch.ones_like(attn.in_proj_a.weight, dtype=torch.float32)
    b_mask = torch.ones_like(attn.in_proj_b.weight, dtype=torch.float32)
    out_mask = torch.ones_like(attn.out_proj.weight, dtype=torch.float32)
    key_dim = geometry["key_dim"]
    for g in remove_groups:
        q = block(g, geometry["q_rows_per_group"])
        k = block(g, geometry["k_rows_per_group"], key_dim)
        v = block(g, geometry["v_rows_per_group"], 2 * key_dim)
        for rows in (q, k, v):
            qkv_mask[rows, :] = 0
            conv_mask[rows, :, :] = 0
        z_mask[block(g, geometry["z_rows_per_group"]), :] = 0
        a_mask[block(g, geometry["a_rows_per_group"]), :] = 0
        b_mask[block(g, geometry["b_rows_per_group"]), :] = 0
        out_mask[:, block(g, geometry["out_cols_per_group"])] = 0
    return {
        f"{prefix}.linear_attn.in_proj_qkv": qkv_mask,
        f"{prefix}.linear_attn.conv1d": conv_mask,
        f"{prefix}.linear_attn.in_proj_z": z_mask,
        f"{prefix}.linear_attn.in_proj_a": a_mask,
        f"{prefix}.linear_attn.in_proj_b": b_mask,
        f"{prefix}.linear_attn.out_proj": out_mask,
    }


def build_weight_masks(model, plan: MaskPlan) -> dict[str, torch.Tensor]:
    config = model.config
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    masks = {}
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        remove_groups = plan.attention.get(layer_idx, set())
        if remove_groups:
            masks.update(kv_group_weight_masks(model, layer_idx, prefix, remove_groups))
        remove_linear_groups = plan.linear_attn.get(layer_idx, set())
        if remove_linear_groups:
            masks.update(
                linear_attention_group_weight_masks(model, layer_idx, prefix, remove_linear_groups)
            )
        remove_neurons = plan.mlp.get(layer_idx, set())
        if remove_neurons:
            masks.update(
                mlp_neuron_weight_masks(
                    prefix,
                    remove_neurons,
                    intermediate_size=intermediate_size,
                    hidden_size=hidden_size,
                )
            )
    return masks
