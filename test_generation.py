import argparse
import asyncio
import json
import os
from typing import Any

import aiohttp
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ENDPOINT = os.getenv("DEFAULT_ENDPOINT")
DEFAULT_PROMPT = "Write a tiny Python function that adds two numbers."


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the deployed Modal vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Base URL for the deployed Modal endpoint.",
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
    args = parser.parse_args()

    if not args.endpoint:
        raise SystemExit(
            "Missing endpoint. Set DEFAULT_ENDPOINT in .env or pass --endpoint."
        )

    endpoint = args.endpoint.rstrip("/")
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
    }

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(base_url=endpoint, timeout=timeout) as session:
        async with session.get("/health") as response:
            response.raise_for_status()
            print(f"Health check: {response.status} OK")

        print("\nModel response:\n")
        async with session.post("/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line.removeprefix("data: ")

                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
