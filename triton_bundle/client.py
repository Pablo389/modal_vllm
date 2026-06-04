from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from triton_bundle.prompts import STRUCTURED_SYSTEM_PROMPT


class StructuredParseError(ValueError):
    """Model response could not be parsed as the expected JSON object."""

    def __init__(self, message: str, *, raw: str, extracted: str) -> None:
        super().__init__(message)
        self.raw = raw
        self.extracted = extracted


def openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint
    return f"{endpoint}/v1"


def extract_json_text(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def parse_structured_bundle(raw: str) -> dict:
    """Parse model output into a bundle dict; raises StructuredParseError on failure."""
    extracted = extract_json_text(raw)
    if not extracted:
        raise StructuredParseError("empty model response", raw=raw, extracted=extracted)
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise StructuredParseError(
            f"invalid JSON: {exc}",
            raw=raw,
            extracted=extracted,
        ) from exc
    if not isinstance(data, dict):
        raise StructuredParseError(
            f"expected JSON object, got {type(data).__name__}",
            raw=raw,
            extracted=extracted,
        )
    return data


def load_json_value(value: str | Path | dict) -> dict:
    data = load_json_any(value)
    if not isinstance(data, dict):
        raise TypeError("structured JSON must decode to an object")
    return data


def load_json_any(value: str | Path | dict | list) -> object:
    if isinstance(value, (dict, list)):
        return value
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(str(value))


def generate_text(
    *,
    endpoint: str,
    api_key: str,
    messages: list[dict[str, str]],
    model: str = "llm",
    temperature: float = 0.2,
    timeout: float = 300.0,
    max_completion_tokens: int | None = None,
    stream: bool = True,
    extra_body: dict[str, Any] | None = None,
) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=openai_base_url(endpoint),
        timeout=timeout,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body

    if stream:
        response = client.chat.completions.create(**kwargs)
        parts: list[str] = []
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                parts.append(content)
        return "".join(parts)

    completion = client.chat.completions.create(**kwargs)
    message = completion.choices[0].message
    return (
        message.content
        or getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )


def generate_structured(
    *,
    endpoint: str,
    api_key: str,
    user_prompt: str,
    json_schema: dict | str | Path,
    model: str = "llm",
    temperature: float = 0.2,
    timeout: float = 300.0,
    max_completion_tokens: int = 8192,
    system_prompt: str = STRUCTURED_SYSTEM_PROMPT,
) -> dict:
    schema = load_json_value(json_schema) if not isinstance(json_schema, dict) else json_schema
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = generate_text(
        endpoint=endpoint,
        api_key=api_key,
        messages=messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_completion_tokens=max_completion_tokens,
        stream=False,
        extra_body={"structured_outputs": {"json": schema}},
    )
    return parse_structured_bundle(raw)
