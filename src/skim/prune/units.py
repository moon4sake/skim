from __future__ import annotations


def kv_group_geometry(model, layer_idx: int) -> dict[str, int]:
    attn = model.model.layers[layer_idx].self_attn
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "kv_group_size": num_heads // num_kv_heads,
        "q_rows_per_head": attn.q_proj.weight.shape[0] // num_heads,
        "o_cols_per_head": attn.o_proj.weight.shape[1] // num_heads,
        "kv_rows_per_head": attn.k_proj.weight.shape[0] // num_kv_heads,
    }


def linear_attention_group_geometry(model, layer_idx: int) -> dict[str, int]:
    attn = model.model.layers[layer_idx].linear_attn
    num_k_heads = int(attn.num_k_heads)
    head_k_dim = int(attn.head_k_dim)
    value_heads_per_group = int(attn.num_v_heads) // num_k_heads
    value_rows_per_group = value_heads_per_group * int(attn.head_v_dim)
    return {
        "num_k_heads": num_k_heads,
        "key_dim": int(attn.key_dim),
        "q_rows_per_group": head_k_dim,
        "k_rows_per_group": head_k_dim,
        "v_rows_per_group": value_rows_per_group,
        "conv_rows_per_group": head_k_dim + head_k_dim + value_rows_per_group,
        "z_rows_per_group": value_rows_per_group,
        "out_cols_per_group": value_rows_per_group,
        "a_rows_per_group": value_heads_per_group,
        "b_rows_per_group": value_heads_per_group,
    }


def mlp_neuron_param_count(model) -> int:
    return int(3 * model.config.hidden_size)


def kv_group_param_count(model, layer_idx: int) -> int:
    attn = model.model.layers[layer_idx].self_attn
    geometry = kv_group_geometry(model, layer_idx)
    q_rows_per_group = geometry["kv_group_size"] * geometry["q_rows_per_head"]
    o_cols_per_group = geometry["kv_group_size"] * geometry["o_cols_per_head"]
    kv_rows_per_head = geometry["kv_rows_per_head"]
    return int(
        q_rows_per_group * attn.q_proj.weight.shape[1]
        + o_cols_per_group * attn.o_proj.weight.shape[0]
        + kv_rows_per_head * attn.k_proj.weight.shape[1]
        + kv_rows_per_head * attn.v_proj.weight.shape[1]
    )


def linear_attention_group_param_count(model, layer_idx: int) -> int:
    attn = model.model.layers[layer_idx].linear_attn
    geometry = linear_attention_group_geometry(model, layer_idx)
    hidden_size = attn.in_proj_qkv.weight.shape[1]
    return int(
        geometry["q_rows_per_group"] * hidden_size
        + geometry["k_rows_per_group"] * hidden_size
        + geometry["v_rows_per_group"] * hidden_size
        + geometry["conv_rows_per_group"] * attn.conv1d.weight.shape[-1]
        + geometry["z_rows_per_group"] * attn.in_proj_z.weight.shape[1]
        + geometry["out_cols_per_group"] * attn.out_proj.weight.shape[0]
        + geometry["a_rows_per_group"] * attn.in_proj_a.weight.shape[1]
        + geometry["b_rows_per_group"] * attn.in_proj_b.weight.shape[1]
    )


def family_param_totals(model) -> dict[str, float]:
    p_mlp = (
        float(mlp_neuron_param_count(model))
        * int(model.config.intermediate_size)
        * int(model.config.num_hidden_layers)
    )
    p_attn = 0.0
    p_linattn = 0.0
    for layer_idx in range(int(model.config.num_hidden_layers)):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, "self_attn"):
            n_groups = kv_group_geometry(model, layer_idx)["num_kv_heads"]
            p_attn += float(kv_group_param_count(model, layer_idx)) * n_groups
        if hasattr(layer, "linear_attn"):
            n_groups = linear_attention_group_geometry(model, layer_idx)["num_k_heads"]
            p_linattn += float(linear_attention_group_param_count(model, layer_idx)) * n_groups
    return {"mlp": p_mlp, "attention": p_attn, "linear_attn": p_linattn}
