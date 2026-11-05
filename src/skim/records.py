from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import random
import tempfile
from pathlib import Path

from tqdm import tqdm

from skim import labels, settings
from skim.config import MAX_SEQ_LEN, N_XLAM, RECORD_CACHE_SHARDS, XLAM_SEED
from skim.data import chat, schema, xlam_loader
from skim.data.xlam_loader import load_xlam
from skim.grammar import xgrammar
from skim.grammar.xgrammar import share_tables
from skim.labels import build_records

RECORD_SOURCES = (labels, schema, chat, xlam_loader, xgrammar)


def xlam_samples(n: int = N_XLAM, seed: int = XLAM_SEED) -> list[dict]:
    samples = load_xlam()
    if n <= 0 or n >= len(samples):
        return samples
    return [samples[i] for i in sorted(random.Random(seed).sample(range(len(samples)), n))]


def cache_fingerprint(tokenizer) -> str:
    digest = hashlib.sha1()
    digest.update(json.dumps(sorted(tokenizer.get_vocab().items())).encode())
    digest.update(str(getattr(tokenizer, "chat_template", "")).encode())
    for module in RECORD_SOURCES:
        digest.update(module.__name__.encode())
        digest.update(inspect.getsource(module).encode())
    return digest.hexdigest()[:12]


def _shard_slice(n: int, shard_index: int, num_shards: int) -> slice:
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard {shard_index} out of range for {num_shards} shards")
    return slice(n * shard_index // num_shards, n * (shard_index + 1) // num_shards)


def record_cache_path(
    tokenizer, n: int, max_seq_len: int, shard_index: int, num_shards: int
) -> Path:
    stem = f"xlam_n{n}_seed{XLAM_SEED}_len{max_seq_len}_{cache_fingerprint(tokenizer)}"
    return settings.RECORD_CACHE_DIR / f"{stem}_shard{shard_index + 1}of{num_shards}.pkl"


def record_shard(
    tokenizer,
    shard_index: int = 0,
    num_shards: int = 1,
    n: int = N_XLAM,
    max_seq_len: int = MAX_SEQ_LEN,
):
    path = record_cache_path(tokenizer, n, max_seq_len, shard_index, num_shards)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    samples = xlam_samples(n)
    samples = samples[_shard_slice(len(samples), shard_index, num_shards)]
    records, stats = build_records(
        tqdm(samples, desc=f"records {shard_index + 1}/{num_shards}"),
        tokenizer,
        max_seq_len,
    )
    share_tables([r.valid_sets for r in records])
    stats.update(shard_index=shard_index, num_shards=num_shards, n=n, max_seq_len=max_seq_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        pickle.dump((records, stats), f, protocol=pickle.HIGHEST_PROTOCOL)
    Path(f.name).replace(path)
    path.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(f"[records] shard {shard_index + 1}/{num_shards}: {stats}", flush=True)
    return records, stats


def iter_record_shards(
    tokenizer,
    num_shards: int = RECORD_CACHE_SHARDS,
    n: int = N_XLAM,
    max_seq_len: int = MAX_SEQ_LEN,
):
    for shard_index in range(num_shards):
        yield record_shard(tokenizer, shard_index, num_shards, n, max_seq_len)


def add_stats(total: dict, stats: dict) -> dict:
    for key, value in stats.items():
        if key not in ("shard_index", "num_shards", "n", "max_seq_len"):
            total[key] = total.get(key, 0) + value
    return total


def xlam_records(
    tokenizer,
    num_shards: int = RECORD_CACHE_SHARDS,
    n: int = N_XLAM,
    max_seq_len: int = MAX_SEQ_LEN,
):
    records, total = [], {}
    for shard_records, stats in iter_record_shards(tokenizer, num_shards, n, max_seq_len):
        records += shard_records
        add_stats(total, stats)
    return records, total
