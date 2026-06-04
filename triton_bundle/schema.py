from __future__ import annotations

import json
from pathlib import Path

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

TRITON_KERNEL_SCHEMA = SCHEMAS_DIR / "triton_kernel.json"
TRITON_VECTOR_ADD_SCHEMA = SCHEMAS_DIR / "triton_vector_add.json"

REQUIRED_IMPORT_LINES = (
    "import torch",
    "import triton",
    "import triton.language as tl",
)


def load_schema(path: str | Path) -> dict:
    schema_path = Path(path)
    if not schema_path.is_absolute() and not schema_path.exists():
        schema_path = SCHEMAS_DIR / schema_path
    return json.loads(schema_path.read_text(encoding="utf-8"))


def default_schema() -> dict:
    return load_schema(TRITON_KERNEL_SCHEMA)
