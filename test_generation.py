import argparse
import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ENDPOINT = os.getenv("DEFAULT_ENDPOINT")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
DEFAULT_PROMPT = "Write a tiny Python function that adds two numbers."


def openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint
    return f"{endpoint}/v1"


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

    print("\nModel response:\n")
    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
