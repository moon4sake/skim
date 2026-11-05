from __future__ import annotations

import re

PYTHON_TYPES = {
    "string": str,
    "any": str,
    "integer": int,
    "float": float,
    "number": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "object": dict,
}


def standardize_string(text: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", text).lower().replace("'", '"')


def _standardize(value):
    return standardize_string(value) if isinstance(value, str) else value


def _answer_type(options: list):
    return next((type(option) for option in options if option != ""), None)


def _type_check(value, options: list, expected: type, nested: type | None) -> tuple[bool, bool]:
    answer_type = _answer_type(options)
    is_variable = answer_type is not None and answer_type is not expected
    if type(value) is expected:
        if nested is None:
            return True, is_variable
        for option in options:
            if not isinstance(option, list) or all(
                _type_check(item, option, nested, None)[0] for item in value
            ):
                return True, is_variable
        return False, is_variable
    return answer_type is not None and type(value) is answer_type, True


def _dict_match(value, options: list) -> bool:
    if not isinstance(value, dict):
        return False
    for option in options:
        if not isinstance(option, dict):
            continue
        if all(
            key in option and _standardize(item) in [_standardize(o) for o in option[key]]
            for key, item in value.items()
        ) and all(key in value or "" in allowed for key, allowed in option.items()):
            return True
    return False


def _list_dict_match(value: list, options: list) -> bool:
    return any(
        (option == "" or isinstance(option, list))
        and len(value) == len(option)
        and all(_dict_match(item, [gold]) for item, gold in zip(value, option))
        for option in options
    )


def _list_match(value: list, options: list) -> bool:
    candidates = [option for option in options if isinstance(option, list)]
    if "" in options:
        candidates.append([])
    return [_standardize(item) for item in value] in [
        [_standardize(item) for item in option] for option in candidates
    ]


def param_match(value, options: list, spec: dict) -> bool:
    kind = spec["type"]
    expected = PYTHON_TYPES[kind]
    nested = None
    if kind in ("array", "tuple"):
        nested = PYTHON_TYPES.get(spec.get("items", {}).get("type"))
    if expected is float and type(value) is int:
        value = float(value)
    valid, is_variable = _type_check(value, options, expected, nested)
    if not valid:
        return False
    if not is_variable:
        if expected is dict:
            return _dict_match(value, options)
        if expected is list and nested is dict:
            return _list_dict_match(value, options)
        if expected is str:
            return standardize_string(value) in [
                standardize_string(option) for option in options if isinstance(option, str)
            ]
        if expected is list:
            return _list_match(value, options)
    return value in options


def call_match(call: dict, function: dict, gold: dict) -> bool:
    arguments = call.get("arguments")
    if call.get("name") != function["name"] or not isinstance(arguments, dict):
        return False
    properties = function["parameters"].get("properties", {})
    if any(param not in arguments for param in function["parameters"].get("required", [])):
        return False
    for param, value in arguments.items():
        if param not in properties or param not in gold:
            return False
        if not param_match(value, gold[param], properties[param]):
            return False
    return all(param in arguments or "" in options for param, options in gold.items())


def full_match(calls: list[dict] | None, functions: list[dict], answers: list[dict]) -> bool:
    if not calls or len(calls) != len(answers):
        return False
    by_name = {function["name"]: function for function in functions}
    used = set()
    for answer in answers:
        ((name, gold),) = answer.items()
        function = by_name.get(name, functions[0] if len(functions) == 1 else None)
        if function is None:
            return False
        index = next(
            (
                i
                for i, call in enumerate(calls)
                if i not in used and call_match(call, function, gold)
            ),
            None,
        )
        if index is None:
            return False
        used.add(index)
    return True
