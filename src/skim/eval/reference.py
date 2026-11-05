from __future__ import annotations

import ast
import re

from skim.data.apibank_loader import apibank_tools


def bfcl_reference(sample: dict) -> tuple[list[dict], list[dict]]:
    return sample["function"], sample["ground_truth"]


def typed_value(value, spec: dict | None):
    if not isinstance(value, str) or spec is None:
        return value
    kind, text = spec.get("type"), value.strip()
    if kind == "integer" and re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if kind == "number" and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", text):
        return float(text)
    if kind == "boolean" and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if kind in ("array", "object"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, list if kind == "array" else dict):
            return parsed
    return value


def apibank_reference(sample: dict) -> tuple[list[dict], list[dict]]:
    tools = {tool["name"]: tool["parameters"]["properties"] for tool in apibank_tools(sample)}
    functions = [
        {"name": name, "parameters": {"properties": properties, "required": []}}
        for name, properties in tools.items()
    ]
    answers = [
        {
            call["name"]: {
                param: [typed_value(value, tools.get(call["name"], {}).get(param))]
                for param, value in call["arguments"].items()
            }
        }
        for call in sample["ground_truth"]
    ]
    return functions, answers
