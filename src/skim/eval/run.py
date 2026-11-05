from __future__ import annotations

import functools
import gc
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from skim.config import EVAL_DECODING_MODES, EVAL_MAX_NEW_TOKENS
from skim.data.apibank_loader import (
    apibank_multiple,
    apibank_tools,
    format_apibank_prompt,
    load_apibank,
)
from skim.data.bfcl_loader import (
    bfcl_multiple,
    bfcl_tools,
    format_bfcl_prompt,
    load_bfcl,
)
from skim.eval.metrics import correct, mode_metrics, score
from skim.eval.reference import apibank_reference, bfcl_reference
from skim.grammar.xgrammar import build_call_grammar, constrained_generate, free_generate


@dataclass(frozen=True)
class Dataset:
    load: Callable[[], list[dict]]
    tools: Callable[[dict], list[dict]]
    multiple: Callable[[dict], bool]
    prompt: Callable[[dict, object], str]
    reference: Callable[[dict], tuple[list[dict], list[dict]]]


DATASETS = {
    "bfcl": Dataset(load_bfcl, bfcl_tools, bfcl_multiple, format_bfcl_prompt, bfcl_reference),
    "apibank": Dataset(
        load_apibank,
        apibank_tools,
        apibank_multiple,
        format_apibank_prompt,
        apibank_reference,
    ),
}


def sample_grammar(dataset: str, sample: dict, tokenizer):
    ds = DATASETS[dataset]
    return build_call_grammar(ds.tools(sample), ds.multiple(sample), tokenizer)


class EvalAborted(RuntimeError):
    pass


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc)
    return (
        isinstance(exc, torch.OutOfMemoryError)
        or "out of memory" in text.lower()
        or "ALLOC_FAILED" in text
    )


def is_cuda_fatal(exc: BaseException) -> bool:
    return any(code in str(exc) for code in ("CUDA error", "CUBLAS_STATUS", "CUDNN_STATUS"))


def eval_samples(dataset: str, eval_limit: int = 0) -> list[dict]:
    samples = DATASETS[dataset].load()
    if 0 < eval_limit < len(samples):

        def key(sample: dict) -> str:
            return hashlib.sha256(f"{dataset}:eval_limit:{sample['id']}".encode()).hexdigest()

        samples = sorted(samples, key=key)[:eval_limit]
    return samples


def generate(model, tokenizer, dataset: str, sample: dict, mode: str, max_new_tokens: int):
    device = next(model.parameters()).device
    prompt = DATASETS[dataset].prompt(sample, tokenizer)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if mode == "grammar":
        grammar = sample_grammar(dataset, sample, tokenizer)
        ids, _, terminated, truncated = constrained_generate(
            model, input_ids, grammar, max_new_tokens
        )
        return ids, {"terminated": terminated, "truncated": truncated}
    ids, ended = free_generate(model, tokenizer, input_ids, max_new_tokens)
    return ids, {"ended": ended}


def _retry_once_on_oom(sample_id: str, run: Callable):
    try:
        return run()
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return run()
    except Exception as exc:
        if is_cuda_oom(exc):
            raise EvalAborted(f"CUDA out of memory on {sample_id}; evaluation aborted") from exc
        raise


def evaluate_sample(model, tokenizer, dataset: str, sample: dict, mode: str, max_new_tokens: int):
    row = {"id": sample["id"], "category": sample["category"], "mode": mode}
    run = functools.partial(generate, model, tokenizer, dataset, sample, mode, max_new_tokens)
    try:
        ids, info = _retry_once_on_oom(sample["id"], run)
    except EvalAborted:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if is_cuda_fatal(exc):
            raise EvalAborted(f"{error} on {sample['id']}; evaluation aborted") from exc
        info = {"terminated": False, "truncated": False} if mode == "grammar" else {"ended": False}
        row.update(raw_output="", n_new_tokens=0, **info, error=error)
        row.update(format_error=False, full_match=False, correct=False)
        return row
    text = tokenizer.decode(ids, skip_special_tokens=True)
    row.update(raw_output=text, n_new_tokens=len(ids), **info, error=None)
    row.update(score(text, *DATASETS[dataset].reference(sample)))
    row["correct"] = correct(row)
    return row


def evaluate_dataset(
    model,
    tokenizer,
    dataset: str,
    *,
    dump_path: str | Path,
    modes: tuple[str, ...] = EVAL_DECODING_MODES,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    eval_limit: int = 0,
) -> dict:
    unknown = set(modes) - set(EVAL_DECODING_MODES)
    if unknown:
        raise ValueError(f"unknown decoding modes {sorted(unknown)}")
    samples = eval_samples(dataset, eval_limit)
    metrics = {}
    with open(dump_path, "w") as dump:
        for mode in modes:
            rows = []
            for sample in tqdm(samples, desc=f"{dataset}_{mode}"):
                row = evaluate_sample(model, tokenizer, dataset, sample, mode, max_new_tokens)
                dump.write(json.dumps(row, ensure_ascii=False) + "\n")
                dump.flush()
                rows.append(row)
            metrics[mode] = mode_metrics(rows, mode)
    return metrics
