from __future__ import annotations

from skim.eval.answer import extract_answer, parse_calls
from skim.eval.match import full_match

HEADLINE = {"grammar": "S", "free": "F"}


def score(text: str, functions: list[dict], answers: list[dict]) -> dict:
    calls = parse_calls(extract_answer(text))
    return {
        "format_error": calls is None,
        "full_match": full_match(calls, functions, answers),
    }


def correct(row: dict) -> bool:
    return bool(row["full_match"]) and (row["mode"] == "grammar" or bool(row["ended"]))


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict], mode: str) -> dict:
    summary = {
        "n": len(rows),
        "n_error": sum(row["error"] is not None for row in rows),
        HEADLINE[mode]: _rate(rows, "correct"),
        "full_match": _rate(rows, "full_match"),
        "format_error": _rate(rows, "format_error"),
    }
    if mode == "free":
        summary["ended"] = _rate(rows, "ended")
    return summary


def mode_metrics(rows: list[dict], mode: str) -> dict:
    categories = sorted({row["category"] for row in rows})
    return {
        "overall": summarize(rows, mode),
        "per_category": {
            category: summarize([row for row in rows if row["category"] == category], mode)
            for category in categories
        },
    }


def headline(metrics: dict) -> dict:
    return {HEADLINE[mode]: metrics[mode]["overall"][HEADLINE[mode]] for mode in metrics}
