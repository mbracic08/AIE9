"""Model utilities for constructing chat LLM clients.

Centralizes configuration of the default chat model and temperature so graphs can
import a single helper without repeating provider-specific wiring.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_chat_model(model_name: str | None = None, *, temperature: float = 0) -> Any:
    """Return a configured LangChain ChatOpenAI client pointed at Fireworks."""
    name = model_name or os.environ.get(
        "FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"
    )
    max_retries = _int_env("LLM_MAX_RETRIES", 8)
    timeout_seconds = _float_env("LLM_REQUEST_TIMEOUT_SECONDS", 90.0)
    return ChatOpenAI(
        model=name,
        temperature=temperature,
        openai_api_key=os.environ["FIREWORKS_API_KEY"],
        openai_api_base=FIREWORKS_BASE_URL,
        max_retries=max_retries,
        request_timeout=timeout_seconds,
    )


def fix_tool_calls(response: AIMessage) -> AIMessage:
    """Fix invalid tool calls caused by models appending extra tokens like <|call|>."""
    if not response.invalid_tool_calls:
        return response

    fixed = list(response.tool_calls)
    remaining_invalid = []

    for tc in response.invalid_tool_calls:
        cleaned = re.sub(r"\s*<\|call\|>\s*$", "", tc["args"])
        try:
            parsed = json.loads(cleaned)
            fixed.append(
                {
                    "name": tc["name"],
                    "args": parsed,
                    "id": tc["id"],
                    "type": "tool_call",
                }
            )
        except (json.JSONDecodeError, TypeError):
            remaining_invalid.append(tc)

    response.tool_calls = fixed
    response.invalid_tool_calls = remaining_invalid
    return response
