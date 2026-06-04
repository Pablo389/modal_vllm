"""Helpers for feeding validated bundles into deepbork predictions.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from triton_bundle.validate import ValidationResult, validate_bundle


def to_prediction_record(
    instruction: str,
    bundle: dict,
    *,
    schema_path: str | Path | None = None,
    expected_wrapper_name: str | None = None,
    expected_signature: str | None = None,
) -> tuple[dict[str, str], ValidationResult]:
    """Validate a bundle and build a deepbork predictions.jsonl row."""
    result = validate_bundle(
        bundle,
        schema_path=schema_path,
        expected_wrapper_name=expected_wrapper_name,
        expected_signature=expected_signature,
    )
    if not result.valid or not result.module:
        return {"instruction": instruction, "predict": ""}, result
    return {"instruction": instruction, "predict": result.module}, result


def write_predictions_jsonl(
    path: str | Path,
    records: list[dict[str, str]],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return out
