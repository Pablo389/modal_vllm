#!/usr/bin/env python3
"""Print kernel/launcher code from a structured out.json file."""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Display code fields from out.json.")
    parser.add_argument("path", nargs="?", default="out.json", help="JSON output file.")
    args = parser.parse_args()

    try:
        with open(args.path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {args.path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

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
        if isinstance(value, list):
            print("\n".join(value))
        else:
            print(value)


if __name__ == "__main__":
    main()
