from __future__ import annotations

import json


def extract_answer(text: str) -> str:
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def _reject_constant(token: str):
    raise ValueError(f"non-JSON constant {token}")


def _is_call(call) -> bool:
    return (
        isinstance(call, dict)
        and call.keys() == {"name", "arguments"}
        and isinstance(call["name"], str)
        and isinstance(call["arguments"], dict)
    )


def parse_calls(answer: str) -> list[dict] | None:
    try:
        parsed = json.loads(answer, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return None
    calls = parsed if isinstance(parsed, list) else [parsed]
    if not all(_is_call(call) for call in calls):
        return None
    return calls
