from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triton_bundle.assemble import assemble_module
from triton_bundle.client import StructuredParseError, generate_structured
from triton_bundle.prompts import repair_user_prompt, structured_user_prompt
from triton_bundle.validate import ValidationResult, validate_bundle


def generate_bundle_with_repair(
    *,
    endpoint: str,
    api_key: str,
    instruction: str,
    wrapper_name: str,
    wrapper_signature: str,
    json_schema: dict | str | Path,
    model: str = "llm",
    temperature: float = 0.1,
    timeout: float = 600.0,
    max_completion_tokens: int = 8192,
    max_attempts: int = 3,
    schema_path: str | Path | None = None,
) -> tuple[dict, ValidationResult, int]:
    """
    Generate a structured bundle and retry with repair prompts until valid or budget exhausted.

    Returns (bundle, validation_result, attempts_used).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    bundle: dict[str, Any] = {}
    result = ValidationResult(valid=False, errors=["no attempts run"])
    user_prompt = structured_user_prompt(
        instruction=instruction,
        wrapper_name=wrapper_name,
        wrapper_signature=wrapper_signature,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            bundle = generate_structured(
                endpoint=endpoint,
                api_key=api_key,
                user_prompt=user_prompt,
                json_schema=json_schema,
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_completion_tokens=max_completion_tokens,
            )
        except StructuredParseError as exc:
            bundle = {}
            snippet = exc.extracted[:1500] if exc.extracted else exc.raw[:1500]
            result = ValidationResult(
                valid=False,
                errors=[
                    str(exc),
                    "Ensure the response is one complete JSON object matching the schema.",
                ],
            )
            if attempt >= max_attempts:
                return bundle, result, attempt
            user_prompt = repair_user_prompt(
                instruction=instruction,
                wrapper_name=wrapper_name,
                wrapper_signature=wrapper_signature,
                previous_bundle={"_parse_error": str(exc), "_raw_snippet": snippet},
                errors=result.errors,
            )
            continue

        result = validate_bundle(
            bundle,
            schema_path=schema_path or json_schema,
            expected_wrapper_name=wrapper_name,
            expected_signature=wrapper_signature,
        )
        if result.valid:
            return bundle, result, attempt

        feedback = result.errors + result.warnings
        if attempt >= max_attempts:
            break
        user_prompt = repair_user_prompt(
            instruction=instruction,
            wrapper_name=wrapper_name,
            wrapper_signature=wrapper_signature,
            previous_bundle=bundle,
            errors=feedback,
        )

    return bundle, result, max_attempts


def save_outputs(
    bundle: dict,
    result: ValidationResult,
    *,
    json_path: str | Path,
    module_path: str | Path | None = None,
) -> None:
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if module_path is None:
        module_path = json_file.with_suffix(".py")
    module_text = result.module or assemble_module(bundle)
    Path(module_path).write_text(module_text, encoding="utf-8")
