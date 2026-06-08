from __future__ import annotations

from enum import Enum

from modal_vllm.grammar.triton_grammar import build_grammar_ebnf, validate_ebnf


class GrammarMode(str, Enum):
    NONE = "none"
    TRITON_FENCED = "triton_fenced"
    TRITON_UNFENCED = "triton_unfenced"


def resolve_http_timeout(timeout: int | float) -> float | None:
    """Map CLI timeout to OpenAI client value; 0 means no limit."""
    if timeout <= 0:
        return None
    return float(timeout)


def openai_base_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def resolve_grammar_ebnf(grammar: GrammarMode) -> str | None:
    if grammar == GrammarMode.NONE:
        return None
    if grammar == GrammarMode.TRITON_FENCED:
        ebnf = build_grammar_ebnf(fenced=True)
    elif grammar == GrammarMode.TRITON_UNFENCED:
        ebnf = build_grammar_ebnf(fenced=False)
    else:
        raise ValueError(f"unknown grammar mode {grammar!r}")
    validate_ebnf(ebnf)
    return ebnf


def generate_text(
    provider: str,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int | float,
    *,
    grammar: GrammarMode = GrammarMode.NONE,
    grammar_ebnf: str | None = None,
) -> str:
    """Generate text via an OpenAI-compatible endpoint with optional xgrammar constraints."""
    if provider == "modal-vllm" and not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT.")

    from openai import OpenAI

    client_timeout = resolve_http_timeout(timeout)
    if provider == "modal-vllm":
        client = OpenAI(
            api_key=api_key,
            base_url=openai_base_url(endpoint),
            timeout=client_timeout,
        )
    elif provider == "openai":
        client = OpenAI(api_key=api_key, timeout=client_timeout)
    else:
        raise ValueError(f"unknown provider {provider!r}")

    if grammar_ebnf is None and grammar != GrammarMode.NONE:
        grammar_ebnf = resolve_grammar_ebnf(grammar)

    request_kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    if grammar_ebnf and provider == "modal-vllm":
        request_kwargs["extra_body"] = {
            "structured_outputs": {"grammar": grammar_ebnf},
        }
    elif grammar_ebnf and provider != "modal-vllm":
        print(
            "warning: xgrammar constrained decoding is only supported with provider modal-vllm; "
            "continuing without grammar constraints",
            flush=True,
        )

    completion = client.chat.completions.create(**request_kwargs)
    message = completion.choices[0].message
    return (
        message.content
        or getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )
