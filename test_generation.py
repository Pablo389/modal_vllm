import argparse
import json
import os
import re
import sys

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ENDPOINT = os.getenv("DEFAULT_ENDPOINT")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
DEFAULT_PROMPT = "Write a tiny Python function that adds two numbers."

STRUCTURED_SYSTEM_PROMPT = (
    "You generate structured JSON only. "
    "Do not wrap output in markdown fences. "
    "Follow the JSON schema exactly. "
    "Each string field must be valid content for that field only."
)


def openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint
    return f"{endpoint}/v1"


def extract_json_text(raw: str) -> str:
    """Strip optional markdown fences and surrounding whitespace."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def load_json(value: str) -> object:
    """Load JSON from either a literal string or a file path."""
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the deployed Modal vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Base URL for the deployed Modal endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--model",
        default="llm",
        help="Served model name. The Modal server exposes this as 'llm'.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (lower is more deterministic for codegen).",
    )
    parser.add_argument(
        "--structured-json",
        help=(
            "JSON schema as literal JSON string or file path. "
            "Passed as extra_body.structured_outputs.json."
        ),
    )
    parser.add_argument(
        "--structured-regex",
        help="Regex constraint passed as extra_body.structured_outputs.regex.",
    )
    parser.add_argument(
        "--structured-choice",
        help=(
            "JSON array string or file path for extra_body.structured_outputs.choice, "
            'for example \'["yes","no"]\'.'
        ),
    )
    parser.add_argument(
        "--structured-grammar",
        help="EBNF grammar string for extra_body.structured_outputs.grammar.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write model response only to this file (no 'Model response:' header).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the 'Model response:' banner (useful with shell redirection).",
    )
    args = parser.parse_args()

    if not args.endpoint:
        raise SystemExit(
            "Missing endpoint. Set DEFAULT_ENDPOINT in .env or pass --endpoint."
        )

    client = OpenAI(
        api_key=args.api_key,
        base_url=openai_base_url(args.endpoint),
        timeout=args.timeout,
    )

    extra_body: dict[str, object] = {}
    structured_outputs: dict[str, object] = {}
    if args.structured_json:
        structured_outputs["json"] = load_json(args.structured_json)
    if args.structured_regex:
        structured_outputs["regex"] = args.structured_regex
    if args.structured_choice:
        structured_choice = load_json(args.structured_choice)
        if not isinstance(structured_choice, list):
            raise SystemExit("--structured-choice must decode to a JSON list.")
        structured_outputs["choice"] = structured_choice
    if args.structured_grammar:
        structured_outputs["grammar"] = args.structured_grammar
    use_structured = bool(structured_outputs)
    if use_structured:
        extra_body["structured_outputs"] = structured_outputs

    messages: list[dict[str, str]] = []
    if use_structured:
        messages.append({"role": "system", "content": STRUCTURED_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": args.prompt})

    # Non-streaming is more reliable for structured JSON payloads.
    request_kwargs: dict[str, object] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "stream": not use_structured,
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    show_banner = not args.quiet and sys.stdout.isatty()
    if show_banner:
        print("\nModel response:\n")

    if use_structured:
        response = client.chat.completions.create(**request_kwargs)
        text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(extract_json_text(text))
        except json.JSONDecodeError as exc:
            print(f"Model returned invalid JSON: {exc}", file=sys.stderr)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Wrote raw text to {args.output}", file=sys.stderr)
            else:
                print(text)
            raise SystemExit(1) from exc

        pretty = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(pretty)
            print(f"Wrote valid JSON to {args.output}", file=sys.stderr)
        if not args.output or sys.stdout.isatty():
            print(pretty)
    else:
        stream = client.chat.completions.create(**request_kwargs)
        parts: list[str] = []
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                parts.append(content)
                if not args.output:
                    print(content, end="", flush=True)

        text = "".join(parts)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
                if text and not text.endswith("\n"):
                    f.write("\n")
            print(f"Wrote response to {args.output}", file=sys.stderr)
        elif text:
            print()


if __name__ == "__main__":
    main()
