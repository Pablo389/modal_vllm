#!/usr/bin/env python3
"""Validate and display structured Triton bundles."""

from __future__ import annotations

import argparse
import json
import sys

from triton_bundle.assemble import assemble_module
from triton_bundle.validate import validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a structured Triton JSON bundle and optionally assemble it."
    )
    parser.add_argument("path", nargs="?", default="out.json", help="Bundle JSON file.")
    parser.add_argument(
        "--schema",
        default="schemas/triton_kernel.json",
        help="JSON schema path for validation.",
    )
    parser.add_argument(
        "--wrapper-name",
        help="Expected TritonBench wrapper function name.",
    )
    parser.add_argument(
        "--wrapper-signature",
        help="Expected wrapper signature for parameter checks.",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Print assembled Python module to stdout.",
    )
    parser.add_argument(
        "-o",
        "--module-output",
        help="Write assembled module to this file.",
    )
    args = parser.parse_args()

    try:
        with open(args.path, encoding="utf-8") as f:
            bundle = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read {args.path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    result = validate_bundle(
        bundle,
        schema_path=args.schema,
        expected_wrapper_name=args.wrapper_name,
        expected_signature=args.wrapper_signature,
    )

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if result.valid:
        print(f"valid: {args.path}", file=sys.stderr)
    else:
        for err in result.errors:
            print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)

    module_text = result.module or assemble_module(bundle)
    if args.module_output:
        with open(args.module_output, "w", encoding="utf-8") as f:
            f.write(module_text)
        print(f"Wrote assembled module to {args.module_output}", file=sys.stderr)
    elif args.assemble:
        print(module_text)


if __name__ == "__main__":
    main()
