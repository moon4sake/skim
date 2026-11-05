from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import xgrammar

from skim.config import MAX_SEQ_LEN
from skim.data.chat import render_prompt, sample_messages, tool_block, user_query
from skim.data.schema import call_schema
from skim.data.xlam_loader import xlam_multiple, xlam_target_calls, xlam_tools
from skim.grammar.xgrammar import (
    ValidSets,
    ValidSetsBuilder,
    build_call_grammar,
    next_bitmask,
    tokenizer_info,
)

CALL_KEYS = {"name", "arguments"}
DROP_REASONS = ("no_target", "schema_invalid", "over_length", "grammar_rejected")


class RecordDropped(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Record:
    sample_id: str
    input_ids: list[int]
    prompt_len: int
    context: list[int]
    decision: list[int]
    format: list[int]
    valid_sets: ValidSets
    tools: list[dict]
    multiple: bool


def valid_sets(grammar: xgrammar.CompiledGrammar, target_ids: list[int]) -> ValidSets:
    vocab_size = grammar.tokenizer_info.vocab_size
    matcher = xgrammar.GrammarMatcher(grammar)
    sets = ValidSetsBuilder(vocab_size)
    for t, token in enumerate(target_ids):
        packed = next_bitmask(matcher, vocab_size)
        if not (0 <= token < vocab_size and packed[token >> 5] >> (token & 31) & 1):
            raise ValueError(f"grammar rejects token {token} at output position {t}")
        sets.add(packed)
        matcher.accept_token(token)
    return sets.build()


@lru_cache(maxsize=4)
def _decoded_vocab(tokenizer) -> list[bytes]:
    return list(tokenizer_info(tokenizer).decoded_vocab)


def token_pieces(tokenizer, ids: list[int]) -> list[bytes]:
    vocab = _decoded_vocab(tokenizer)
    return [vocab[i] if 0 <= i < len(vocab) else b"" for i in ids]


def _ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _string_end(text: str, i: int) -> int:
    i += 1
    while text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return i


def _scan(text: str, i: int, path: tuple, events: list) -> int:
    i = _ws(text, i)
    if text[i] == "{":
        i = _ws(text, i + 1)
        if text[i] == "}":
            return i + 1
        while True:
            if text[i] != '"':
                raise ValueError(f"expected a key at {i}")
            end = _string_end(text, i)
            key = text[i + 1 : end]
            events.append(("key", path + (key,), i + 1, end))
            i = _ws(text, end + 1)
            if text[i] != ":":
                raise ValueError(f"expected ':' at {i}")
            i = _ws(text, _scan(text, i + 1, path + (key,), events))
            if text[i] == "}":
                return i + 1
            if text[i] != ",":
                raise ValueError(f"expected ',' or '}}' at {i}")
            i = _ws(text, i + 1)
    if text[i] == "[":
        events.append(("punct", path, i, i + 1))
        i = _ws(text, i + 1)
        n = 0
        while text[i] != "]":
            i = _ws(text, _scan(text, i, path + (n,), events))
            n += 1
            if text[i] == ",":
                events.append(("punct", path, i, i + 1))
                i = _ws(text, i + 1)
            elif text[i] != "]":
                raise ValueError(f"expected ',' or ']' at {i}")
        events.append(("punct", path, i, i + 1))
        return i + 1
    if text[i] == '"':
        end = _string_end(text, i)
        events.append(("value", path, i + 1, end))
        return end + 1
    end = i
    while end < len(text) and text[end] not in ",]} \t\r\n":
        end += 1
    if end == i:
        raise ValueError(f"expected a JSON value at {i}")
    events.append(("value", path, i, end))
    return end


def json_events(text: str) -> list[tuple]:
    events: list[tuple] = []
    try:
        _scan(text, 0, (), events)
    except IndexError as exc:
        raise ValueError("incomplete JSON") from exc
    return events


def _covered(n: int, spans) -> np.ndarray:
    covered = np.zeros(n + 1, dtype=bool)
    for start, end in spans:
        covered[start:end] = True
    return covered


def decision_spans(text: str) -> list[tuple[int, int]]:
    call_depth = 1 if text[_ws(text, 0) : _ws(text, 0) + 1] == "[" else 0
    spans = []
    for kind, path, start, end in json_events(text):
        if kind == "punct" and path != ():
            continue
        if kind == "key" and len(path) == call_depth + 1 and path[-1] in CALL_KEYS:
            continue
        if end > start:
            spans.append((start, end))
    return spans


def output_categories(vsets: ValidSets, pieces: list[bytes]) -> tuple[list[int], list[int]]:
    text = b"".join(pieces).decode("latin-1")
    covered = _covered(len(text), decision_spans(text))
    sizes = vsets.sizes()
    decision, fmt, start = [], [], 0
    for t, piece in enumerate(pieces):
        end = start + len(piece)
        (decision if sizes[t] > 1 and covered[start:end].any() else fmt).append(t)
        start = end
    return decision, fmt


def _context_spans(block: str) -> list[tuple[int, int]]:
    spans = []
    for kind, path, start, end in json_events(block):
        if len(path) < 3 or path[1] != "function":
            continue
        if kind == "value" and len(path) == 3 and path[2] in ("name", "description"):
            spans.append((start, end))
        elif path[2] == "parameters" and len(path) >= 5:
            if kind == "key" and path[-2] == "properties":
                spans.append((start, end))
            elif kind == "value" and path[-1] == "description":
                spans.append((start, end))
    return spans


def context_positions(tokenizer, prompt_text: str, tools: list[dict], query: str) -> list[int]:
    spans = []
    block = tool_block(tools)
    at = prompt_text.find(block)
    if at >= 0:
        spans += [(at + s, at + e) for s, e in _context_spans(block)]
    query = query.strip()
    at = prompt_text.rfind(query) if query else -1
    if at >= 0:
        spans.append((at, at + len(query)))
    covered = _covered(len(prompt_text), [s for s in spans if s[1] > s[0]])
    offsets = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    return [p for p, (s, e) in enumerate(offsets["offset_mapping"]) if covered[s:e].any()]


def predicting_state(positions) -> list[int]:
    return [int(p) - 1 for p in positions]


def _ordered(value, schema):
    if isinstance(schema, dict) and schema.get("anyOf"):
        name = value.get("name") if isinstance(value, dict) else None
        variants = schema["anyOf"]
        schema = next(
            (v for v in variants if v.get("properties", {}).get("name", {}).get("const") == name),
            variants[0],
        )
    if isinstance(value, list):
        item = schema.get("items") if isinstance(schema, dict) else None
        return [_ordered(v, item) for v in value]
    if not isinstance(value, dict):
        return value
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    out = {k: _ordered(value[k], properties[k]) for k in properties if k in value}
    out.update({k: _ordered(v, None) for k, v in value.items() if k not in out})
    return out


def serialize_target(calls: list[dict], multiple: bool, schema: dict) -> str:
    payload = _ordered(calls if multiple else calls[0], schema)
    return json.dumps(payload)


def relabel(record: Record, target_ids: list[int], vsets: ValidSets, tokenizer) -> Record:
    decision, fmt = output_categories(vsets, token_pieces(tokenizer, target_ids))
    p = record.prompt_len
    return replace(
        record,
        input_ids=record.input_ids[:p] + [int(t) for t in target_ids],
        decision=[p + t for t in decision],
        format=[p + t for t in fmt],
        valid_sets=vsets,
    )


def build_record(sample: dict, tokenizer, max_seq_len: int = MAX_SEQ_LEN) -> Record:
    tools, multiple, calls = xlam_tools(sample), xlam_multiple(sample), xlam_target_calls(sample)
    if not calls:
        raise RecordDropped("no_target")
    try:
        schema = call_schema(tools, multiple)
        grammar = build_call_grammar(tools, multiple, tokenizer)
    except Exception as exc:
        raise RecordDropped("schema_invalid") from exc
    messages = sample_messages(sample)
    prompt_text = render_prompt(tokenizer, messages, tools)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(
        serialize_target(calls, multiple, schema), add_special_tokens=False
    )
    target_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(target_ids) > max_seq_len:
        raise RecordDropped("over_length")
    try:
        vsets = valid_sets(grammar, target_ids)
    except ValueError as exc:
        raise RecordDropped("grammar_rejected") from exc
    prompt = Record(
        sample_id=str(sample.get("id", "")),
        input_ids=prompt_ids,
        prompt_len=len(prompt_ids),
        context=context_positions(tokenizer, prompt_text, tools, user_query(messages)),
        decision=[],
        format=[],
        valid_sets=vsets,
        tools=tools,
        multiple=multiple,
    )
    return relabel(prompt, target_ids, vsets, tokenizer)


def build_records(samples: list[dict], tokenizer, max_seq_len: int = MAX_SEQ_LEN):
    records, stats = [], {"seen": 0, "kept": 0, **{reason: 0 for reason in DROP_REASONS}}
    for sample in samples:
        stats["seen"] += 1
        try:
            records.append(build_record(sample, tokenizer, max_seq_len))
            stats["kept"] += 1
        except RecordDropped as dropped:
            stats[dropped.reason] += 1
    stats["decision_positions"] = sum(len(r.decision) for r in records)
    stats["format_positions"] = sum(len(r.format) for r in records)
    stats["context_positions"] = sum(len(r.context) for r in records)
    return records, stats
