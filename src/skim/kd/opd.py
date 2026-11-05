from __future__ import annotations

import torch

from skim.config import (
    KD_SCHEDULE,
    MAX_SEQ_LEN,
    OPD_WARMUP_FRAC,
    ROLLOUT_MAX_NEW_TOKENS,
)
from skim.grammar.xgrammar import build_call_grammar, constrained_generate
from skim.labels import Record, relabel

SCHEDULES = ("hybrid", "off_policy")
FALLBACKS = ("compile_failed", "empty", "truncated", "over_length")


def phase(
    step: int,
    total_steps: int,
    schedule: str = KD_SCHEDULE,
    warmup_frac: float = OPD_WARMUP_FRAC,
) -> str:
    if schedule == "off_policy" or step < int(warmup_frac * total_steps):
        return "off_policy"
    return "on_policy"


def rollout_counts() -> dict[str, int]:
    return {"rollouts": 0, **{f"fallback_{reason}": 0 for reason in FALLBACKS}}


def _fallback(record: Record, counts: dict, reason: str) -> Record:
    counts[f"fallback_{reason}"] += 1
    return record


def on_policy_record(
    student,
    record: Record,
    tokenizer,
    counts: dict,
    device,
    max_new_tokens: int = ROLLOUT_MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Record:
    try:
        grammar = build_call_grammar(record.tools, record.multiple, tokenizer)
    except (RuntimeError, ValueError):
        return _fallback(record, counts, "compile_failed")
    prompt = torch.tensor([record.input_ids[: record.prompt_len]], device=device)
    ids, vsets, terminated, _ = constrained_generate(student, prompt, grammar, max_new_tokens)
    if not ids:
        return _fallback(record, counts, "empty")
    if not terminated:
        return _fallback(record, counts, "truncated")
    if record.prompt_len + len(ids) > max_seq_len:
        return _fallback(record, counts, "over_length")
    counts["rollouts"] += 1
    return relabel(record, ids, vsets, tokenizer)


def on_policy_batch(
    student,
    records: list[Record],
    tokenizer,
    counts: dict,
    device,
    max_new_tokens: int = ROLLOUT_MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN,
) -> list[Record]:
    was_training = student.training
    student.eval()
    try:
        return [
            on_policy_record(student, r, tokenizer, counts, device, max_new_tokens, max_seq_len)
            for r in records
        ]
    finally:
        student.train(was_training)
