"""Offline tests for triton_bundle assembly and validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from triton_bundle.assemble import assemble_module
from triton_bundle.validate import validate_bundle

OUT_JSON = Path(__file__).resolve().parent.parent / "out.json"


class TritonBundleTests(unittest.TestCase):
    def test_assemble_injects_required_imports(self) -> None:
        bundle = {
            "imports": "import torch",
            "kernel_code": "@triton.jit\ndef k():\n    pass",
            "launcher_code": "def f():\n    pass",
        }
        module = assemble_module(bundle)
        self.assertIn("import triton", module)
        self.assertIn("import triton.language as tl", module)

    def test_validate_div_example(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        result = validate_bundle(
            bundle,
            expected_wrapper_name="div",
            expected_signature="div(input, other, *, rounding_mode=None, out=None) -> Tensor",
        )
        self.assertTrue(result.valid, result.errors)
        self.assertIsNotNone(result.module)
        self.assertIn("def div(", result.module or "")

    def test_validate_rejects_missing_wrapper(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["launcher_code"] = "def wrong():\n    pass"
        result = validate_bundle(
            bundle,
            expected_wrapper_name="div",
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("div" in err for err in result.errors))

    def test_validate_rejects_int_grid(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["kernel_name"] = "_div_kernel"
        bundle["kernel_code"] = (
            "@triton.jit\ndef _div_kernel(input_ptr, other_ptr, output_ptr, n, mode, BLOCK_SIZE: tl.constexpr):\n"
            "    pass\n"
        )
        bundle["launcher_code"] = (
            "def div(input, other, *, rounding_mode=None, out=None):\n"
            "    n = input.numel()\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE\n"
            "    _div_kernel[grid](input.data_ptr(), other.data_ptr(), out.data_ptr(), n, 0, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(any("tuple" in err.lower() for err in result.errors))

    def test_validate_rejects_bare_tensors_at_launch(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["kernel_name"] = "_div_kernel"
        bundle["kernel_code"] = (
            "@triton.jit\ndef _div_kernel(input_ptr, other_ptr, output_ptr, n, mode, BLOCK_SIZE: tl.constexpr):\n"
            "    pass\n"
        )
        bundle["launcher_code"] = (
            "def div(input, other, *, rounding_mode=None, out=None):\n"
            "    n = input.numel()\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = (triton.cdiv(n, BLOCK_SIZE),)\n"
            "    _div_kernel[grid](input, other, out, n, 0, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(any("data_ptr" in err for err in result.errors))

    def test_validate_rejects_tl_trunc(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["kernel_code"] = (
            "@triton.jit\ndef _div_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):\n"
            "    x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))\n"
            "    tl.store(x_ptr, tl.trunc(x))\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(any("tl.trunc" in err for err in result.errors))

    def test_validate_rejects_runtime_rounding_mode_in_kernel(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["kernel_code"] = (
            "@triton.jit\ndef _div_kernel(input_ptr, other_ptr, output_ptr, n_elements, rounding_mode, BLOCK_SIZE: tl.constexpr):\n"
            "    if rounding_mode == 1:\n"
            "        pass\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(
            any("tl.constexpr" in err for err in result.errors),
            result.errors,
        )

    def test_validate_rejects_data_ptr_without_cast(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["kernel_code"] = (
            "@triton.jit\ndef _div_kernel(input_ptr, other_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):\n"
            "    x = tl.load(input_ptr + tl.arange(0, BLOCK_SIZE))\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(any("tl.cast" in err for err in result.errors))

    def test_validate_rejects_missing_broadcast(self) -> None:
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["launcher_code"] = (
            "def div(input, other, *, rounding_mode=None, out=None):\n"
            "    n_elements = input.numel()\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)\n"
            "    _div_kernel[grid](input.contiguous().data_ptr(), other.contiguous().data_ptr(), out.data_ptr(), n_elements, rounding_mode=0, BLOCK_SIZE=BLOCK_SIZE)\n"
            "    return out\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(any("broadcast_tensors" in err for err in result.errors))

    def test_validate_rejects_user_bad_div_launcher(self) -> None:
        """Regression: model output with int grid and tensor args."""
        bundle = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        bundle["kernel_name"] = "div_kernel"
        bundle["kernel_code"] = (
            "@triton.jit\ndef div_kernel(x_ptr, y_ptr, output_ptr, n_elements, rounding_mode: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    offsets = tl.arange(0, BLOCK_SIZE)\n"
            "    tl.store(output_ptr + offsets, tl.load(x_ptr + offsets) / tl.load(y_ptr + offsets))\n"
        )
        bundle["launcher_code"] = (
            "def div(input, other, *, rounding_mode=None, out=None):\n"
            "    n_elements = input.numel()\n"
            "    BLOCK_SIZE = 1024\n"
            "    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE\n"
            "    div_kernel[grid](input, other, out, n_elements, 0, BLOCK_SIZE)\n"
            "    return out\n"
        )
        result = validate_bundle(bundle, expected_wrapper_name="div")
        self.assertFalse(result.valid)
        self.assertTrue(any("tuple" in err.lower() for err in result.errors))
        self.assertTrue(any("data_ptr" in err for err in result.errors))


if __name__ == "__main__":
    unittest.main()
