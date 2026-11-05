from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import xgrammar

from skim.data.schema import call_schema


@lru_cache(maxsize=4)
def tokenizer_info(tokenizer) -> xgrammar.TokenizerInfo:
    return xgrammar.TokenizerInfo.from_huggingface(tokenizer)


@lru_cache(maxsize=4)
def _compiler(tokenizer) -> xgrammar.GrammarCompiler:
    return xgrammar.GrammarCompiler(tokenizer_info(tokenizer))


def compile_schema(schema: dict, tokenizer) -> xgrammar.CompiledGrammar:
    return _compiler(tokenizer).compile_json_schema(json.dumps(schema, separators=(",", ":")))


def build_call_grammar(tools: list[dict], multiple: bool, tokenizer) -> xgrammar.CompiledGrammar:
    return compile_schema(call_schema(tools, multiple), tokenizer)


def stop_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        ids.add(tokenizer.convert_tokens_to_ids(token))
    return sorted(i for i in ids if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id)


def _unpack(packed: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.unpackbits(packed.view(np.uint8), bitorder="little")[:vocab_size].astype(bool)


@dataclass
class ValidSets:
    table: np.ndarray
    index: np.ndarray
    vocab_size: int

    def sizes(self) -> np.ndarray:
        counts = np.unpackbits(self.table.view(np.uint8), axis=1).sum(axis=1)
        return counts[self.index]

    def mask(self, width: int | None = None, device=None) -> torch.Tensor:
        packed = torch.from_numpy(self.table.view(np.int32)[self.index]).to(device)
        shifts = torch.arange(32, device=packed.device, dtype=torch.int32)
        bits = ((packed.unsqueeze(-1) >> shifts) & 1).bool().flatten(1)[:, : self.vocab_size]
        width = self.vocab_size if width is None else int(width)
        if width > bits.shape[1]:
            bits = torch.nn.functional.pad(bits, (0, width - bits.shape[1]), value=False)
        return bits[:, :width]


class ValidSetsBuilder:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.rows: dict[bytes, int] = {}
        self.table: list[np.ndarray] = []
        self.index: list[int] = []

    def add(self, packed: np.ndarray) -> int:
        key = packed.tobytes()
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = len(self.table)
            self.table.append(packed)
        self.index.append(row)
        return row

    def build(self) -> ValidSets:
        words = (self.vocab_size + 31) // 32
        table = np.stack(self.table) if self.table else np.zeros((0, words), dtype=np.uint32)
        return ValidSets(table, np.asarray(self.index, dtype=np.int32), self.vocab_size)


def share_tables(valid_sets: list[ValidSets]) -> None:
    if not valid_sets:
        return
    builder = ValidSetsBuilder(valid_sets[0].vocab_size)
    for vs in valid_sets:
        remap = np.asarray([builder.add(row) for row in vs.table], dtype=np.int32)
        vs.index = remap[vs.index] if len(remap) else vs.index
    table = builder.build().table
    for vs in valid_sets:
        vs.table = table


def next_bitmask(matcher: xgrammar.GrammarMatcher, vocab_size: int) -> np.ndarray:
    bitmask = xgrammar.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return bitmask[0].numpy().view(np.uint32).copy()


def constrained_generate(
    model, input_ids: torch.Tensor, grammar: xgrammar.CompiledGrammar, max_new_tokens: int
):
    vocab_size = grammar.tokenizer_info.vocab_size
    matcher = xgrammar.GrammarMatcher(grammar)
    sets = ValidSetsBuilder(vocab_size)
    ids: list[int] = []
    current, past = input_ids, None
    while len(ids) < max_new_tokens and not matcher.is_terminated():
        with torch.no_grad():
            out = model(current, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        packed = next_bitmask(matcher, vocab_size)
        allowed = torch.from_numpy(_unpack(packed, vocab_size)).to(logits.device)
        masked = torch.full_like(logits, -float("inf"))
        masked[:vocab_size] = logits[:vocab_size].masked_fill(~allowed, -float("inf"))
        token = int(masked.argmax())
        if not matcher.accept_token(token):
            break
        sets.add(packed)
        ids.append(token)
        current = torch.tensor([[token]], device=input_ids.device)
    terminated = matcher.is_terminated()
    return ids, sets.build(), terminated, not terminated and len(ids) == max_new_tokens


def free_generate(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int):
    stops = stop_token_ids(tokenizer)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            eos_token_id=stops,
            pad_token_id=tokenizer.eos_token_id,
        )
    ids = [int(t) for t in out[0, input_ids.shape[1] :]]
    return ids, bool(ids) and ids[-1] in stops
