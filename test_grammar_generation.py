import argparse
import os

from dotenv import load_dotenv

from modal_vllm.client.generate import GrammarMode, generate_text


load_dotenv()

DEFAULT_ENDPOINT = os.getenv("DEFAULT_ENDPOINT")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "llm")

TRITON_PROMPT = """You are an expert in Triton programming.

Write a minimal Triton module for vector addition with:
- import torch, triton, triton.language as tl
- one @triton.jit kernel
- one wrapper def with return

Wrap the entire module in one ```python ... ``` fenced code block."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Modal vLLM generation with optional Triton DSL xgrammar constraints."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Base URL for the deployed Modal endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Served model name.",
    )
    parser.add_argument(
        "--prompt",
        default=TRITON_PROMPT,
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--grammar",
        choices=[mode.value for mode in GrammarMode if mode != GrammarMode.NONE],
        default=GrammarMode.TRITON_FENCED.value,
        help="Triton DSL grammar mode. Omit with --no-grammar.",
    )
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Disable xgrammar constrained decoding.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Request timeout in seconds. Use 0 for no limit.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.endpoint:
        raise SystemExit(
            "Missing endpoint. Set DEFAULT_ENDPOINT in .env or pass --endpoint."
        )

    grammar = GrammarMode.NONE if args.no_grammar else GrammarMode(args.grammar)
    if grammar != GrammarMode.NONE:
        print(f"using grammar mode: {grammar.value}", flush=True)
        print(
            "note: xgrammar is slow; Modal vLLM deploy must use function timeout well above 600s",
            flush=True,
        )

    messages = [{"role": "user", "content": args.prompt}]
    print("\nModel response:\n", flush=True)
    text = generate_text(
        provider="modal-vllm",
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0.2,
        timeout=args.timeout,
        grammar=grammar,
    )
    print(text)


if __name__ == "__main__":
    main()
