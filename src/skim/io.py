from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path_or_name: str):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(path_or_name: str, device: str, dtype: torch.dtype | None = None):
    dtype = dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(path_or_name, trust_remote_code=True, dtype=dtype)
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return model.to(device).eval()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
