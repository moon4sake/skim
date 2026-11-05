from __future__ import annotations

import json
import os
from urllib.request import urlretrieve

from skim.data.chat import render_prompt, sample_messages
from skim.data.schema import strict_json_schema
from skim.settings import APIBANK_DIR

APIBANK_L1_URL = (
    "https://raw.githubusercontent.com/MadeAgents/Hammer/main/data/test/original/apibank_l1.json"
)
_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "list": "array",
    "dict": "object",
    "bool": "boolean",
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
}


def load_apibank() -> list[dict]:
    path = APIBANK_DIR / "apibank_l1.json"
    if not path.exists():
        APIBANK_DIR.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        urlretrieve(APIBANK_L1_URL, partial)
        partial.replace(path)
    with open(path) as f:
        raw = json.load(f)
    samples = []
    for i, item in enumerate(raw):
        messages = [
            {
                "role": "assistant" if t.get("role") == "agent" else t.get("role", "user"),
                "content": t.get("content", ""),
            }
            for t in item.get("query", [])
        ]
        samples.append(
            {
                "id": f"apibank_l1_{i}",
                "question": [messages],
                "function": item.get("tools", []),
                "ground_truth": item.get("answers", []),
                "category": "apibank_l1",
            }
        )
    return samples


def _is_optional(info: dict) -> bool:
    description = str(info.get("description", "")).lower()
    return (
        info.get("default") is not None
        or info.get("optional") is True
        or "optional" in description
        or "only needed for" in description
    )


def _apibank_parameters(tool: dict) -> dict:
    properties, required = {}, []
    for name, info in tool.get("parameters", {}).items():
        prop = {"type": _TYPE_MAP.get(str(info.get("type", "str")).lower().strip(), "string")}
        if "description" in info:
            prop["description"] = info["description"]
        properties[name] = prop
        if not _is_optional(info):
            required.append(name)
    return strict_json_schema({"type": "object", "properties": properties, "required": required})


def apibank_tools(sample: dict) -> list[dict]:
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": _apibank_parameters(tool),
        }
        for tool in sample["function"]
    ]


def apibank_multiple(sample: dict) -> bool:
    return False


def format_apibank_prompt(sample: dict, tokenizer) -> str:
    return render_prompt(tokenizer, sample_messages(sample), apibank_tools(sample))
