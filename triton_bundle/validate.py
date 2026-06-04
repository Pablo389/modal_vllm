from __future__ import annotations

import ast
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator

from triton_bundle.assemble import assemble_module
from triton_bundle.schema import load_schema, TRITON_KERNEL_SCHEMA


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    module: str | None = None


def _parse_signature_names(signature: str) -> tuple[str, list[str], list[str]]:
    """Extract function name, positional param names, and keyword-only param names."""
    head = signature.split("->", 1)[0].strip()
    match = re.match(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*$", head, re.DOTALL)
    if not match:
        raise ValueError(f"cannot parse signature: {signature!r}")

    func_name = match.group(1)
    params_blob = match.group(2).strip()
    if not params_blob:
        return func_name, [], []

    positional: list[str] = []
    kwonly: list[str] = []
    section = "positional"
    for part in params_blob.split(","):
        token = part.strip()
        if not token:
            continue
        if token == "*":
            section = "kwonly"
            continue
        if token.startswith("*"):
            section = "kwonly"
            token = token.lstrip("*").strip()
        name = token.split("=", 1)[0].strip()
        if not name:
            continue
        if section == "positional":
            positional.append(name)
        else:
            kwonly.append(name)
    return func_name, positional, kwonly


def _has_triton_jit_decorator(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute):
            if isinstance(dec.value, ast.Name) and dec.value.id == "triton" and dec.attr == "jit":
                return True
        if isinstance(dec, ast.Name) and dec.id == "jit":
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "triton" and func.attr == "jit":
                    return True
            if isinstance(func, ast.Name) and func.id == "jit":
                return True
    return False


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _kernel_referenced_in_launcher(launcher: ast.FunctionDef, kernel_name: str) -> bool:
    for node in ast.walk(launcher):
        if isinstance(node, ast.Name) and node.id == kernel_name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == kernel_name:
            return True
    return False


def _is_tuple_grid(value: ast.expr) -> bool:
    if isinstance(value, ast.Tuple) and value.elts:
        return True
    return False


def _grid_assignments(launcher: ast.FunctionDef) -> dict[str, ast.expr]:
    grids: dict[str, ast.expr] = {}
    for node in ast.walk(launcher):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "grid":
                grids[target.id] = node.value
    return grids


def _check_grid_launch(
    launcher: ast.FunctionDef,
    kernel_name: str,
    errors: list[str],
) -> None:
    """Require tuple grid for kernel[grid] launches."""
    grid_vars = _grid_assignments(launcher)
    for node in ast.walk(launcher):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not isinstance(value, ast.Name) or value.id != kernel_name:
            continue
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
            errors.append(
                f"kernel {kernel_name}: grid must be a tuple, not an integer literal"
            )
            return
        if isinstance(slice_node, ast.Name):
            var_name = slice_node.id
            assigned = grid_vars.get(var_name)
            if assigned is not None and not _is_tuple_grid(assigned):
                errors.append(
                    f"kernel {kernel_name}: grid variable {var_name!r} must be a tuple, "
                    f"e.g. grid = (triton.cdiv(n_elements, BLOCK_SIZE),)"
                )
                return
            if assigned is None:
                for assign in ast.walk(launcher):
                    if not isinstance(assign, ast.Assign):
                        continue
                    for target in assign.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            if isinstance(assign.value, ast.BinOp) and not _is_tuple_grid(assign.value):
                                errors.append(
                                    f"kernel {kernel_name}: grid {var_name!r} looks like int division; "
                                    "wrap in a tuple: grid = (triton.cdiv(...),)"
                                )
                                return


def _kernel_ptr_params(kernel_fn: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in kernel_fn.args.args if arg.arg.endswith("_ptr")]


def _kernel_constexpr_params(kernel_fn: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for arg in kernel_fn.args.args:
        if arg.annotation and isinstance(arg.annotation, ast.Attribute):
            if (
                isinstance(arg.annotation.value, ast.Name)
                and arg.annotation.value.id == "tl"
                and arg.annotation.attr == "constexpr"
            ):
                names.append(arg.arg)
    return names


def _check_kernel_launch_args(
    launcher: ast.FunctionDef,
    kernel_fn: ast.FunctionDef,
    kernel_name: str,
    errors: list[str],
) -> None:
    ptr_params = _kernel_ptr_params(kernel_fn)
    constexpr_params = _kernel_constexpr_params(kernel_fn)
    if not ptr_params and not constexpr_params:
        return

    for node in ast.walk(launcher):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Subscript):
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != kernel_name:
            continue

        for index, arg in enumerate(node.args):
            if index >= len(kernel_fn.args.args):
                break
            param = kernel_fn.args.args[index]
            if param.arg in ptr_params:
                if isinstance(arg, ast.Name):
                    errors.append(
                        f"kernel {kernel_name}: pass .data_ptr() for {param.arg}, "
                        f"not bare tensor {arg.id!r}"
                    )
                elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                    if arg.func.id not in {"int", "float"}:
                        errors.append(
                            f"kernel {kernel_name}: argument for {param.arg} "
                            "should be tensor.data_ptr()"
                        )

        constexpr_kw = {kw.arg for kw in node.keywords if kw.arg}
        for name in constexpr_params:
            if name not in constexpr_kw:
                pos_count = len(node.args)
                constexpr_index = next(
                    (i for i, a in enumerate(kernel_fn.args.args) if a.arg == name),
                    None,
                )
                if constexpr_index is not None and constexpr_index >= pos_count:
                    errors.append(
                        f"kernel {kernel_name}: pass {name} as keyword "
                        f"{name}={name}"
                    )


def _is_constexpr_arg(arg: ast.arg) -> bool:
    ann = arg.annotation
    return (
        ann is not None
        and isinstance(ann, ast.Attribute)
        and isinstance(ann.value, ast.Name)
        and ann.value.id == "tl"
        and ann.attr == "constexpr"
    )


def _params_in_if_test_compares(fn: ast.FunctionDef) -> set[str]:
    """Kernel params compared with == inside if tests (not mask bounds like offsets < n_elements)."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for sub in ast.walk(node.test):
            if not isinstance(sub, ast.Compare):
                continue
            for expr in (sub.left, *sub.comparators):
                if isinstance(expr, ast.Name):
                    names.add(expr.id)
    return names


def _check_kernel_constexpr_branches(
    kernel_fn: ast.FunctionDef,
    kernel_name: str,
    errors: list[str],
) -> None:
    """Integer args used in if/== branches must be tl.constexpr (Triton compile requirement)."""
    compared = _params_in_if_test_compares(kernel_fn)
    for arg in kernel_fn.args.args:
        if arg.arg in compared and not _is_constexpr_arg(arg):
            errors.append(
                f"kernel {kernel_name}: parameter {arg.arg!r} is used in if/== branches "
                "but is not declared as name: tl.constexpr"
            )


def _launcher_uses_broadcast(launcher: ast.FunctionDef) -> bool:
    for node in ast.walk(launcher):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "broadcast_tensors"
            ):
                return True
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                if (
                    isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "torch"
                    and value.func.attr == "broadcast_tensors"
                ):
                    return True
    return False


def _check_launcher_broadcast(
    wrapper_fn: ast.FunctionDef,
    kernel_name: str,
    errors: list[str],
) -> None:
    param_names = {a.arg for a in wrapper_fn.args.args} | {a.arg for a in wrapper_fn.args.kwonlyargs}
    if "input" in param_names and "other" in param_names and not _launcher_uses_broadcast(wrapper_fn):
        errors.append(
            f"launcher {wrapper_fn.name}: call torch.broadcast_tensors(input, other) "
            f"before computing n_elements and launching {kernel_name}"
        )


def _check_kernel_source(kernel_code: str, kernel_name: str, errors: list[str]) -> None:
    if re.search(r"\btl\.trunc\b", kernel_code) or re.search(r"\btl\.math\.trunc\b", kernel_code):
        errors.append(
            f"kernel {kernel_name}: do not use tl.trunc; use floor/ceil or where"
        )
    if re.search(r"\bif\s+rounding_mode\s*==", kernel_code) and not re.search(
        r"rounding_mode\s*:\s*tl\.constexpr", kernel_code
    ):
        errors.append(
            f"kernel {kernel_name}: rounding_mode used in if branches must be "
            "rounding_mode: tl.constexpr"
        )
    if re.search(r"tl\.load\(\w+_ptr\s*\+", kernel_code) and not re.search(
        r"tl\.cast\(\w+_ptr\s*,\s*tl\.pointer_type\(",
        kernel_code,
    ):
        errors.append(
            f"kernel {kernel_name}: cast each *_ptr from data_ptr() with "
            "tl.cast(ptr, tl.pointer_type(tl.float32)) before tl.load/tl.store"
        )


def validate_bundle(
    bundle: dict,
    *,
    schema_path: str | Path | None = None,
    expected_wrapper_name: str | None = None,
    expected_signature: str | None = None,
) -> ValidationResult:
    """Validate JSON schema, assemble module, and run semantic checks."""
    errors: list[str] = []
    warnings: list[str] = []

    schema_file = schema_path or TRITON_KERNEL_SCHEMA
    schema = load_schema(schema_file)
    validator = Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(bundle), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.path) or "(root)"
        errors.append(f"schema: {path}: {err.message}")

    wrapper_name = expected_wrapper_name or bundle.get("wrapper_name")
    kernel_name = bundle.get("kernel_name")

    if wrapper_name and bundle.get("wrapper_name") and bundle["wrapper_name"] != wrapper_name:
        errors.append(
            f"wrapper_name mismatch: expected {wrapper_name!r}, "
            f"got {bundle['wrapper_name']!r}"
        )

    module: str | None = None
    if not errors:
        kernel_code = bundle.get("kernel_code", "")
        if kernel_name and kernel_code:
            _check_kernel_source(kernel_code, kernel_name, errors)

        try:
            module = assemble_module(bundle)
        except ValueError as exc:
            errors.append(str(exc))
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        try:
            tree = ast.parse(module)
        except SyntaxError as exc:
            errors.append(f"syntax error in assembled module: {exc}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        kernel_fn = None
        if kernel_name:
            kernel_fn = _find_function(tree, kernel_name)
            if kernel_fn is None:
                errors.append(f"kernel function {kernel_name!r} not found in assembled module")
            elif not _has_triton_jit_decorator(kernel_fn):
                errors.append(f"kernel {kernel_name!r} missing @triton.jit decorator")

        wrapper_fn = None
        if wrapper_name:
            wrapper_fn = _find_function(tree, wrapper_name)
            if wrapper_fn is None:
                errors.append(f"wrapper function {wrapper_name!r} not found in launcher_code")
            elif expected_signature:
                try:
                    sig_name, sig_pos, sig_kw = _parse_signature_names(expected_signature)
                except ValueError as exc:
                    warnings.append(str(exc))
                    sig_name, sig_pos, sig_kw = wrapper_name, [], []
                else:
                    if sig_name != wrapper_name:
                        errors.append(
                            f"expected wrapper {wrapper_name!r}, signature declares {sig_name!r}"
                        )
                    actual_pos = [arg.arg for arg in wrapper_fn.args.args]
                    actual_kw = [arg.arg for arg in wrapper_fn.args.kwonlyargs]
                    if sig_pos and actual_pos != sig_pos:
                        errors.append(
                            f"wrapper positional params mismatch: expected {sig_pos}, got {actual_pos}"
                        )
                    if sig_kw and actual_kw != sig_kw:
                        errors.append(
                            f"wrapper keyword-only params mismatch: expected {sig_kw}, got {actual_kw}"
                        )

            if wrapper_fn and kernel_name and not _kernel_referenced_in_launcher(wrapper_fn, kernel_name):
                errors.append(f"launcher does not reference kernel {kernel_name!r}")
            if wrapper_fn and kernel_name:
                _check_grid_launch(wrapper_fn, kernel_name, errors)
                _check_launcher_broadcast(wrapper_fn, kernel_name, errors)
            if kernel_fn and kernel_name:
                _check_kernel_constexpr_branches(kernel_fn, kernel_name, errors)
            if wrapper_fn and kernel_fn and kernel_name:
                _check_kernel_launch_args(wrapper_fn, kernel_fn, kernel_name, errors)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(module)
            tmp_path = tmp.name
        try:
            compile(module, tmp_path, "exec")
        except SyntaxError as exc:
            errors.append(f"compile error: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        module=module,
    )
