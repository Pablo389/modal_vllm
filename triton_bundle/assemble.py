from __future__ import annotations

import re

from triton_bundle.schema import REQUIRED_IMPORT_LINES

_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\s+\S+")


def _split_import_lines(imports: str) -> list[str]:
    lines: list[str] = []
    for raw in imports.splitlines():
        line = raw.strip()
        if line and _IMPORT_LINE.match(line):
            lines.append(line)
    return lines


def _merge_imports(imports: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for line in REQUIRED_IMPORT_LINES:
        if line not in seen:
            seen.add(line)
            ordered.append(line)

    for line in _split_import_lines(imports):
        if line not in seen:
            seen.add(line)
            ordered.append(line)

    return ordered


def _strip_leading_imports(code: str) -> str:
    """Remove import lines from kernel/launcher snippets if the model duplicated them."""
    lines = code.splitlines()
    while lines and _IMPORT_LINE.match(lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def assemble_module(bundle: dict) -> str:
    """Build a self-contained Python module from a structured Triton bundle."""
    imports = _merge_imports(bundle.get("imports", ""))
    kernel_code = _strip_leading_imports(bundle.get("kernel_code", ""))
    launcher_code = _strip_leading_imports(bundle.get("launcher_code", ""))

    if not kernel_code:
        raise ValueError("bundle missing kernel_code")
    if not launcher_code:
        raise ValueError("bundle missing launcher_code")

    parts = ["\n".join(imports), "", kernel_code, "", launcher_code]
    return "\n".join(parts).strip() + "\n"
