from __future__ import annotations

import math

import numpy as np
import torch

from skim.config import BATCH, EPOCHS, GRAD_ACCUM, SEED
from skim.labels import Record


def epoch_order(n: int, epoch: int, seed: int = SEED) -> list[int]:
    return np.random.default_rng([seed, epoch]).permutation(n).tolist()


def epoch_batches(records: list[Record], epoch: int, batch: int = BATCH, seed: int = SEED):
    order = epoch_order(len(records), epoch, seed)
    return [[records[i] for i in order[s : s + batch]] for s in range(0, len(order), batch)]


def optimizer_steps(
    n_records: int,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    grad_accum: int = GRAD_ACCUM,
) -> int:
    return math.ceil(epochs * math.ceil(n_records / batch) / grad_accum)


def collate(records: list[Record], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r.input_ids) for r in records)
    input_ids = torch.full((len(records), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), width), dtype=torch.long)
    for b, r in enumerate(records):
        input_ids[b, : len(r.input_ids)] = torch.tensor(r.input_ids)
        attention_mask[b, : len(r.input_ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


def output_rows(records: list[Record], width: int) -> tuple[int, list[slice]]:
    start = min(r.prompt_len for r in records) - 1
    rows = [slice(r.prompt_len - 1 - start, len(r.input_ids) - 1 - start) for r in records]
    return width - start, rows
