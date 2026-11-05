from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from skim.data.schema import strict_json_schema
from skim.settings import XLAM_DIR

REPO_ID = "Salesforce/xlam-function-calling-60k"
DATA_FILE = "xlam_function_calling_60k.json"

_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def _download_if_needed() -> Path:
    XLAM_DIR.mkdir(parents=True, exist_ok=True)
    local_path = XLAM_DIR / DATA_FILE
    if local_path.exists():
        return local_path
    try:
        return Path(
            hf_hub_download(
                REPO_ID,
                DATA_FILE,
                repo_type="dataset",
                local_dir=str(XLAM_DIR),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download Salesforce/xlam-function-calling-60k. "
            "Make sure the Hugging Face account on this machine has accepted "
            "the gated dataset terms and is authenticated, or place "
            f"`{DATA_FILE}` under `{XLAM_DIR}`."
        ) from exc


def _maybe_parse_json(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalize_type_name(type_name: str) -> str:
    return _TYPE_MAP.get(type_name.lower().strip(), "string")


def _type_spec_to_schema(type_name: str) -> dict:
    spec = str(type_name).strip()
    lowered = spec.lower()

    if spec.endswith("[]"):
        inner = spec[:-2].strip()
        return {"type": "array", "items": _type_spec_to_schema(inner) if inner else {}}

    open_idx = spec.find("[")
    if open_idx != -1 and spec.endswith("]"):
        container = lowered[:open_idx].strip()
        inner = spec[open_idx + 1 : -1].strip()
        if container in {"list", "array", "sequence", "tuple", "set"}:
            first = inner.split(",")[0].strip() if inner else ""
            return {"type": "array", "items": _type_spec_to_schema(first) if first else {}}
        if container in {"dict", "mapping", "object"}:
            return {"type": "object"}
        if container == "optional":
            return _type_spec_to_schema(inner) if inner else {"type": "string"}
        return {"type": _normalize_type_name(container)}

    return {"type": _normalize_type_name(spec)}


def _parse_xlam_type_spec(type_name: str) -> tuple[dict, bool]:
    parts = [part.strip() for part in str(type_name).split(",") if part.strip()]
    if not parts:
        return {"type": "string"}, False
    is_optional = any(part.lower() == "optional" for part in parts[1:])
    schema = _type_spec_to_schema(parts[0])
    return schema, is_optional


def _normalize_json_schema(schema):
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                result[key] = _normalize_type_name(value)
            else:
                result[key] = _normalize_json_schema(value)
        return result
    if isinstance(schema, list):
        return [_normalize_json_schema(item) for item in schema]
    return schema


def _xlam_params_to_json_schema(raw_params) -> dict:
    raw_params = _maybe_parse_json(raw_params)
    if not isinstance(raw_params, dict):
        return {"type": "object", "properties": {}}

    if raw_params.get("type") == "object" or "properties" in raw_params:
        schema = _normalize_json_schema(raw_params)
        if "type" not in schema:
            schema["type"] = "object"
        schema.setdefault("properties", {})
        return strict_json_schema(schema)

    properties = {}
    required = []
    for param_name, param_info in raw_params.items():
        if not isinstance(param_info, dict):
            properties[param_name] = {"type": "string"}
            required.append(param_name)
            continue

        param_schema, is_optional = _parse_xlam_type_spec(str(param_info.get("type", "string")))
        properties[param_name] = dict(param_schema)
        if properties[param_name].get("type") == "integer":
            default = param_info.get("default")
            try:
                if default is not None and float(str(default)) != int(float(str(default))):
                    properties[param_name]["type"] = "number"
            except (ValueError, TypeError):
                pass
        if "description" in param_info:
            properties[param_name]["description"] = str(param_info["description"])
        has_default = "default" in param_info and param_info.get("default") is not None
        default_required = not (is_optional or has_default)
        if bool(param_info.get("required", default_required)):
            required.append(param_name)

    return strict_json_schema(
        {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    )


def _normalize_messages(query) -> list[dict]:
    query = _maybe_parse_json(query)
    if isinstance(query, str):
        return [{"role": "user", "content": query}]

    if isinstance(query, list):
        messages = []
        for turn in query:
            if isinstance(turn, dict) and "content" in turn:
                role = str(turn.get("role", "user"))
                if role == "agent":
                    role = "assistant"
                messages.append({"role": role, "content": str(turn.get("content", ""))})
            else:
                messages.append({"role": "user", "content": str(turn)})
        return messages

    return [{"role": "user", "content": json.dumps(query, ensure_ascii=False)}]


def load_xlam() -> list[dict]:
    path = _download_if_needed()
    with open(path) as f:
        raw_data = json.load(f)

    samples = []
    for index, item in enumerate(raw_data):
        query = _maybe_parse_json(item.get("query", ""))
        tools = _maybe_parse_json(item.get("tools", []))
        answers = _maybe_parse_json(item.get("answers", []))

        if isinstance(tools, dict):
            tools = [tools]
        if not isinstance(tools, list):
            tools = []
        if isinstance(answers, dict):
            answers = [answers]
        if not isinstance(answers, list):
            answers = []

        item_id = item.get("id", index)
        samples.append(
            {
                "id": f"xlam_fc_60k_{item_id}",
                "question": [_normalize_messages(query)],
                "function": tools,
                "ground_truth": answers,
            }
        )

    return samples


def xlam_tools(sample: dict) -> list[dict]:
    return [
        {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": _xlam_params_to_json_schema(tool.get("parameters", {})),
        }
        for tool in sample["function"]
    ]


def xlam_multiple(sample: dict) -> bool:
    return len(sample.get("ground_truth", [])) > 1


def xlam_target_calls(sample: dict) -> list[dict]:
    return [
        {"name": call.get("name", ""), "arguments": call.get("arguments", {})}
        for call in sample.get("ground_truth", [])
    ]
