from __future__ import annotations

import json

TOOL_INSTRUCTION = (
    "You have access to the following functions.\n\n"
    "{tools}\n\n"
    "When a function call is needed, reply only with raw JSON and no markdown, XML, or prose.\n"
    "For one call, use exactly this shape:\n"
    '{{"name":"function_name","arguments":{{"argument_name":"argument_value"}}}}\n'
    "For multiple calls, reply with a JSON array of objects in that same shape.\n"
    "Do not include keys that are not declared by the selected function schema."
)


def tool_block(tools: list[dict]) -> str:
    wrapped = [{"type": "function", "function": tool} for tool in tools]
    return json.dumps(wrapped, ensure_ascii=True, indent=2)


def render_prompt(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    messages = [dict(message) for message in messages]
    instruction = TOOL_INSTRUCTION.format(tools=tool_block(tools))
    if messages and str(messages[0].get("role", "")).lower() == "system":
        content = str(messages[0].get("content", "")).strip()
        messages[0]["content"] = f"{instruction}\n\n{content}" if content else instruction
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def sample_messages(sample: dict) -> list[dict]:
    return sample["question"][0] if sample["question"] else []


def user_query(messages: list[dict]) -> str:
    query = ""
    for turn in messages:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role not in {"assistant", "system", "tool"} and content:
            query = content
    return query
