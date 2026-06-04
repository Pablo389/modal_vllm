#!/usr/bin/env python3
"""CLI for structured and free-form generation against the Modal vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from triton_bundle.assemble import assemble_module
from triton_bundle.client import (
    StructuredParseError,
    extract_json_text,
    generate_structured,
    generate_text,
    load_json_any,
    load_json_value,
    parse_structured_bundle,
)
from triton_bundle.pipeline import generate_bundle_with_repair, save_outputs
from triton_bundle.prompts import STRUCTURED_SYSTEM_PROMPT, structured_user_prompt
from triton_bundle.validate import validate_bundle


load_dotenv()

DEFAULT_ENDPOINT = os.getenv("DEFAULT_ENDPOINT")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
DEFAULT_PROMPT = "Write a tiny Python function that adds two numbers."


def _run_structured(args: argparse.Namespace, user_prompt: str) -> dict:
    json_only = (
        args.structured_json
        and not args.structured_regex
        and not args.structured_choice
        and not args.structured_grammar
    )
    if json_only:
        return generate_structured(
            endpoint=args.endpoint,
            api_key=args.api_key,
            user_prompt=user_prompt,
            json_schema=args.structured_json,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_completion_tokens=args.max_tokens,
        )

    structured_outputs: dict[str, object] = {}
    if args.structured_json:
        structured_outputs["json"] = load_json_value(args.structured_json)
    if args.structured_regex:
        structured_outputs["regex"] = args.structured_regex
    if args.structured_choice:
        structured_choice = load_json_any(args.structured_choice)
        if not isinstance(structured_choice, list):
            raise SystemExit("--structured-choice must decode to a JSON list.")
        structured_outputs["choice"] = structured_choice
    if args.structured_grammar:
        structured_outputs["grammar"] = args.structured_grammar

    raw = generate_text(
        endpoint=args.endpoint,
        api_key=args.api_key,
        messages=[
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_completion_tokens=args.max_tokens,
        stream=False,
        extra_body={"structured_outputs": structured_outputs},
    )
    try:
        return parse_structured_bundle(raw)
    except StructuredParseError as exc:
        print(f"Model returned invalid JSON: {exc}", file=sys.stderr)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(exc.raw)
        else:
            print(exc.raw)
        raise SystemExit(1) from exc


def _write_assembled(args: argparse.Namespace, bundle: dict) -> None:
    module_path = args.module_output
    if not module_path and args.output:
        module_path = str(args.output) + ".py"

    if args.validate:
        result = validate_bundle(
            bundle,
            schema_path=args.structured_json,
            expected_wrapper_name=args.wrapper_name,
            expected_signature=args.wrapper_signature,
        )
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if not result.valid:
            for err in result.errors:
                print(f"error: {err}", file=sys.stderr)
            raise SystemExit(1)
        module_text = result.module or assemble_module(bundle)
    else:
        module_text = assemble_module(bundle)

    if module_path:
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(module_text)
        print(f"Wrote assembled module to {module_path}", file=sys.stderr)
    elif sys.stdout.isatty():
        print("\n--- assembled module ---\n")
        print(module_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the deployed Modal vLLM OpenAI-compatible endpoint."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Modal serve base URL.")
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt.")
    parser.add_argument("--model", default="llm", help="Served model name.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout (seconds).")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max completion tokens per request.",
    )
    parser.add_argument(
        "--structured-json",
        help="JSON schema as literal JSON or file path (enables XGrammar structured output).",
    )
    parser.add_argument("--structured-regex", help="Regex constraint for structured_outputs.")
    parser.add_argument(
        "--structured-choice",
        help='JSON list for structured_outputs.choice, e.g. \'["yes","no"]\'.',
    )
    parser.add_argument("--structured-grammar", help="EBNF grammar for structured_outputs.")
    parser.add_argument(
        "--wrapper-name",
        help="Expected TritonBench wrapper name (used with --validate / --repair-attempts).",
    )
    parser.add_argument(
        "--wrapper-signature",
        help="Expected wrapper signature for semantic validation.",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="Regenerate on validation failure up to N times (needs --wrapper-name and --wrapper-signature).",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="After structured generation, assemble a Python module.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run semantic validation (implies --assemble).",
    )
    parser.add_argument("-o", "--output", help="Write JSON bundle to this file.")
    parser.add_argument(
        "--module-output",
        help="Write assembled Python module here (default: <output>.py when assembling).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the 'Model response:' banner.",
    )
    args = parser.parse_args()

    if not args.endpoint:
        raise SystemExit("Missing endpoint. Set DEFAULT_ENDPOINT in .env or pass --endpoint.")

    use_structured = any(
        (
            args.structured_json,
            args.structured_regex,
            args.structured_choice,
            args.structured_grammar,
        )
    )

    if args.repair_attempts > 1:
        if not (args.wrapper_name and args.wrapper_signature and args.structured_json):
            raise SystemExit(
                "--repair-attempts requires --wrapper-name, --wrapper-signature, and --structured-json"
            )
        args.validate = True
        args.assemble = True

    user_prompt = args.prompt
    if use_structured and args.wrapper_name and args.wrapper_signature:
        user_prompt = structured_user_prompt(
            instruction=args.prompt,
            wrapper_name=args.wrapper_name,
            wrapper_signature=args.wrapper_signature,
        )

    show_banner = not args.quiet and sys.stdout.isatty()
    if show_banner:
        print("\nModel response:\n")

    if use_structured:
        if args.repair_attempts > 1:
            bundle, result, used = generate_bundle_with_repair(
                endpoint=args.endpoint,
                api_key=args.api_key,
                instruction=args.prompt,
                wrapper_name=args.wrapper_name,
                wrapper_signature=args.wrapper_signature,
                json_schema=args.structured_json,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_completion_tokens=args.max_tokens,
                max_attempts=args.repair_attempts,
                schema_path=args.structured_json,
            )
            print(f"attempts used: {used}/{args.repair_attempts}", file=sys.stderr)
            if not result.valid:
                for err in result.errors:
                    print(f"error: {err}", file=sys.stderr)
                for warning in result.warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                raise SystemExit(1)
            if args.output:
                save_outputs(
                    bundle,
                    result,
                    json_path=args.output,
                    module_path=args.module_output or (str(args.output) + ".py"),
                )
                print(f"Wrote valid JSON to {args.output}", file=sys.stderr)
            pretty = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
            if not args.output or sys.stdout.isatty():
                print(pretty)
            return

        bundle = _run_structured(args, user_prompt)
        pretty = json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(pretty)
            print(f"Wrote valid JSON to {args.output}", file=sys.stderr)
        if not args.output or sys.stdout.isatty():
            print(pretty)
        if args.assemble or args.validate:
            _write_assembled(args, bundle)
        return

    raw = generate_text(
        endpoint=args.endpoint,
        api_key=args.api_key,
        messages=[{"role": "user", "content": user_prompt}],
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_completion_tokens=args.max_tokens,
        stream=True,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(raw)
            if raw and not raw.endswith("\n"):
                f.write("\n")
        print(f"Wrote response to {args.output}", file=sys.stderr)
    elif raw:
        print(raw, end="" if raw.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
