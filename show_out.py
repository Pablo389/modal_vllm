#!/usr/bin/env python3
"""Print code fields from a structured Triton bundle."""

from __future__ import annotations

import argparse
import json
import sys

from triton_bundle.assemble import assemble_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Display code fields from a structured bundle.")
    parser.add_argument("path", nargs="?", default="out.json", help="Bundle JSON file.")
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Print full assembled Python module instead of individual fields.",
    )
    args = parser.parse_args()

    try:
        with open(args.path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read {args.path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.assemble:
        try:
            print(assemble_module(data))
        except ValueError as exc:
            print(f"assemble error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    for label, key in (
        ("IMPORTS", "imports"),
        ("KERNEL", "kernel_code"),
        ("LAUNCHER", "launcher_code"),
        ("QUICK TEST", "quick_test_code"),
    ):
        value = data.get(key)
        if not value:
            continue
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n")
        print(value)


if __name__ == "__main__":
    main()
