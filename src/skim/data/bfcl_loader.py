from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

from skim.data.chat import render_prompt, sample_messages
from skim.data.schema import strict_json_schema
from skim.settings import BFCL_DIR

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
CATEGORIES = ["simple", "multiple", "parallel"]
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array"}


def _download(path_in_repo: str) -> str:
    local_path = BFCL_DIR / path_in_repo
    if local_path.exists():
        return str(local_path)
    BFCL_DIR.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(REPO_ID, path_in_repo, repo_type="dataset", local_dir=str(BFCL_DIR))


def _normalize_bfcl_types(schema):
    if isinstance(schema, list):
        return [_normalize_bfcl_types(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            if value != "any":
                result[key] = _TYPE_MAP.get(value, value)
        else:
            result[key] = _normalize_bfcl_types(value)
    return result


def load_bfcl_category(category: str) -> list[dict]:
    with open(_download(f"BFCL_v3_{category}.json")) as f:
        questions = [json.loads(line) for line in f]
    with open(_download(f"possible_answer/BFCL_v3_{category}.json")) as f:
        answers = {row["id"]: row["ground_truth"] for row in map(json.loads, f)}
    return [
        {
            "id": q["id"],
            "question": q["question"],
            "function": q["function"],
            "ground_truth": answers.get(q["id"], []),
            "category": category,
        }
        for q in questions
    ]


def load_bfcl() -> list[dict]:
    return [sample for category in CATEGORIES for sample in load_bfcl_category(category)]


def bfcl_tools(sample: dict) -> list[dict]:
    return [
        {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": strict_json_schema(_normalize_bfcl_types(func.get("parameters", {}))),
        }
        for func in sample["function"]
    ]


def bfcl_multiple(sample: dict) -> bool:
    return "parallel" in sample.get("category", "simple")


def format_bfcl_prompt(sample: dict, tokenizer) -> str:
    return render_prompt(tokenizer, sample_messages(sample), bfcl_tools(sample))
