from __future__ import annotations

from copy import deepcopy


def strict_json_schema(schema):
    if isinstance(schema, list):
        return [strict_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in deepcopy(schema).items():
        nested = {"properties", "patternProperties", "$defs", "definitions"}
        if key in nested and isinstance(value, dict):
            result[key] = {name: strict_json_schema(child) for name, child in value.items()}
        else:
            result[key] = strict_json_schema(value)
    if result.get("type") == "object" or "properties" in result:
        result.setdefault("additionalProperties", "properties" not in result)
    return result


def call_schema(tools: list[dict], multiple: bool) -> dict:
    if not tools:
        raise ValueError("no tools")
    calls = [
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": tool["name"]},
                "arguments": tool["parameters"],
            },
            "required": ["name", "arguments"],
        }
        for tool in tools
    ]
    one = calls[0] if len(calls) == 1 else {"anyOf": calls}
    return strict_json_schema({"type": "array", "items": one, "minItems": 1} if multiple else one)
