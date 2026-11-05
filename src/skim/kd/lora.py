from __future__ import annotations

import torch
from torch import nn

from skim.config import LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS
from skim.io import load_model
from skim.prune.masks import MaskApplier


def quantize_nf4(model, device: str):
    import bitsandbytes as bnb
    from peft import prepare_model_for_kbit_training

    compute_dtype = (
        torch.bfloat16
        if str(device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    def swap(module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if (
                isinstance(child, nn.Linear)
                and not isinstance(child, bnb.nn.Linear4bit)
                and "lm_head" not in full
            ):
                new = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type="nf4",
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data.to(torch.float16).contiguous(),
                    requires_grad=False,
                    quant_type="nf4",
                )
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.data.to(compute_dtype), requires_grad=False)
                setattr(module, name, new)
            else:
                swap(child, full)

    swap(model)
    model.config.use_cache = False
    model.to(device)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    for p in model.parameters():
        if p.is_floating_point() and p.dtype == torch.float32:
            p.data = p.data.to(compute_dtype)
    for buf in model.buffers():
        if buf.is_floating_point() and buf.dtype == torch.float32:
            buf.data = buf.data.to(compute_dtype)
    return model


def add_lora(model):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=list(LORA_TARGETS),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def lora_student(model_name: str, masks: dict[str, torch.Tensor], device: str):
    model = load_model(model_name, "cpu", dtype=torch.bfloat16)
    MaskApplier.apply(model, masks)
    MaskApplier.assert_applied(model, masks)
    return add_lora(quantize_nf4(model, device))


def merge_lora(student):
    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit

    base = student.get_base_model() if hasattr(student, "get_base_model") else student

    def dequantize(layer):
        return dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.float16)

    def dense(layer, weight):
        new = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
        new.weight = nn.Parameter(
            weight.detach().to("cpu", torch.float16).contiguous(), requires_grad=False
        )
        if layer.bias is not None:
            new.bias = nn.Parameter(
                layer.bias.data.detach().to("cpu", torch.float16), requires_grad=False
            )
        return new

    def replace(module) -> None:
        for name, child in list(module.named_children()):
            is_lora = (
                hasattr(child, "base_layer")
                and hasattr(child, "lora_A")
                and hasattr(child, "lora_B")
            )
            if is_lora and isinstance(child.base_layer, bnb.nn.Linear4bit):
                weight = dequantize(child.base_layer)
                for adapter in list(child.lora_A.keys()):
                    a = child.lora_A[adapter].weight.data.to(weight.device, torch.float16)
                    b = child.lora_B[adapter].weight.data.to(weight.device, torch.float16)
                    weight = weight + (b @ a) * child.scaling[adapter]
                setattr(module, name, dense(child.base_layer, weight))
            elif isinstance(child, bnb.nn.Linear4bit):
                setattr(module, name, dense(child, dequantize(child)))
            else:
                replace(child)

    replace(base)
    base = base.to("cpu")
    for p in base.parameters():
        if p.is_floating_point() and p.dtype != torch.float16:
            p.data = p.data.to(torch.float16)
    return base.eval()
