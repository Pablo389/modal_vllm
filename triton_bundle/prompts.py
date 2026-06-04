from __future__ import annotations

import json

LAUNCHER_RULES = """
launcher_code rules (mandatory):
- Call torch.broadcast_tensors(input, other) when both are tensors and shapes may differ.
- Launch grid MUST be a tuple, e.g. grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
- Pass tensor data as .data_ptr() pointers matching kernel *_ptr parameters.
- Pass runtime integers (n_elements, mode codes) as positional args.
- Pass tl.constexpr values as keywords only, e.g. ROUNDING_MODE=mode_code, BLOCK_SIZE=BLOCK_SIZE.
- Use input.contiguous() and other.contiguous() before .data_ptr().
"""

KERNEL_RULES = """
kernel_code rules (mandatory):
- @triton.jit function only; no import lines inside kernel_code.
- Use tl.program_id(0).
- Do not use tl.trunc; implement trunc with tl.floor/tl.where/tl.ceil.
- Kernel pointer args end with _ptr; launcher passes tensor.data_ptr() (int64).
  Cast each *_ptr before load/store: ptr = tl.cast(ptr, tl.pointer_type(tl.float32)).
- Then tl.load(ptr + offsets, mask=mask) and tl.store(ptr + offsets, ...).
- Any integer used in `if name == <int>:` branches MUST be declared as `name: tl.constexpr`
  (e.g. rounding_mode: tl.constexpr). Never use runtime `if rounding_mode == 1` without tl.constexpr.
- n_elements may stay a normal int for mask=offsets < n_elements only.
"""

FEW_SHOT_KERNEL = """
Example kernel skeleton (adapt names):
@triton.jit
def _op_kernel(x_ptr, y_ptr, out_ptr, n_elements, rounding_mode: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    x_ptr = tl.cast(x_ptr, tl.pointer_type(tl.float32))
    y_ptr = tl.cast(y_ptr, tl.pointer_type(tl.float32))
    out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x / y
    if rounding_mode == 1:
        out = tl.where(out >= 0, tl.floor(out), tl.ceil(out))
    elif rounding_mode == 2:
        out = tl.floor(out)
    tl.store(out_ptr + offsets, out, mask=mask)
"""

FEW_SHOT_LAUNCHER = """
Example launcher (adapt names; div uses rounding_mode trunc=1, floor=2):
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    input, other = torch.broadcast_tensors(input, other)
    if out is None:
        out = torch.empty_like(input)
    mode_code = 0
    if rounding_mode == "trunc":
        mode_code = 1
    elif rounding_mode == "floor":
        mode_code = 2
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _op_kernel[grid](
        input.contiguous().data_ptr(),
        other.contiguous().data_ptr(),
        out.data_ptr(),
        n_elements,
        rounding_mode=mode_code,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
"""

STRUCTURED_SYSTEM_PROMPT = (
    "You generate structured JSON only. "
    "Do not wrap output in markdown fences. "
    "Follow the JSON schema exactly. "
    "Each string field must be valid Python source for that field only. "
    "The imports field must include: import torch, import triton, "
    "import triton.language as tl. "
    "kernel_code must define a @triton.jit kernel only (no imports). "
    "launcher_code must define the wrapper function with the exact name and "
    "signature requested in the user message. "
    "Use integer codes for kernel constexpr parameters, not Python strings. "
    + KERNEL_RULES
    + LAUNCHER_RULES
)


def structured_user_prompt(
    *,
    instruction: str,
    wrapper_name: str,
    wrapper_signature: str,
    include_few_shot: bool = True,
) -> str:
    """Build a user prompt aligned with deepbork / TritonBench operator specs."""
    parts = [
        instruction.strip(),
        "",
        "Return JSON matching the schema.",
        f"Set wrapper_name to {wrapper_name!r}.",
        f"The launcher_code must define this exact wrapper signature:",
        f"  {wrapper_signature.strip()}",
        LAUNCHER_RULES.strip(),
    ]
    if include_few_shot:
        parts.extend(
            [
                "",
                "Reference kernel pattern:",
                FEW_SHOT_KERNEL.strip(),
                "",
                "Reference launcher pattern:",
                FEW_SHOT_LAUNCHER.strip(),
            ]
        )
    parts.append("Put operation-specific hints (dtype, axis, shapes) in metadata.extra.")
    return "\n".join(parts)


def repair_user_prompt(
    *,
    instruction: str,
    wrapper_name: str,
    wrapper_signature: str,
    previous_bundle: dict,
    errors: list[str],
) -> str:
    """Prompt for regenerating a bundle after semantic validation failures."""
    error_block = "\n".join(f"- {err}" for err in errors)
    return (
        f"{instruction.strip()}\n\n"
        f"Fix the previous structured output. Validation errors:\n"
        f"{error_block}\n\n"
        f"Previous JSON (fix these fields, keep correct parts):\n"
        f"{json.dumps(previous_bundle, ensure_ascii=False)}\n\n"
        f"wrapper_name must be {wrapper_name!r}.\n"
        f"launcher_code must define: {wrapper_signature.strip()}\n"
        f"{LAUNCHER_RULES.strip()}\n"
        f"{KERNEL_RULES.strip()}\n"
        f"Reference kernel pattern:\n{FEW_SHOT_KERNEL.strip()}\n"
        f"Reference launch pattern:\n{FEW_SHOT_LAUNCHER.strip()}"
    )
