# Modal vLLM Inference

Minimal repo for serving **Qwen3-Coder** with vLLM on Modal through an OpenAI-compatible API, plus a **`triton_bundle`** library for structured Triton kernel generation (XGrammar), assembly, and semantic validation — ready for integration with [deepbork](../deepbork).

This example serves `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` on an **H100** GPU using Modal Volumes for Hugging Face and vLLM caches.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install local dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Then edit `.env` and set your deployed Modal endpoint. The test client uses the
OpenAI Python library, which requires an API key value. For the unauthenticated
Modal endpoint in this repo, `EMPTY` is fine:

```bash
DEFAULT_ENDPOINT=https://your-workspace-name--example-vllm-inference-serve.modal.run
OPENAI_API_KEY=EMPTY
```

Authenticate Modal:

```bash
modal setup
```

## Test

Run the local entrypoint. Modal will start the remote vLLM server, run a health check, and send a sample chat request:

```bash
modal run vllm_inference.py
```

## Deploy

```bash
modal deploy vllm_inference.py
```

Useful routes:

- `/health` — server health
- `/docs` — Swagger UI
- `/v1/chat/completions` — OpenAI-compatible chat

## Structured Triton bundles (`triton_bundle`)

The `triton_bundle` package turns vLLM **structured_outputs** (XGrammar) into modules that deepbork can evaluate.

```text
prompt + JSON schema  →  vLLM XGrammar  →  bundle JSON
                                              ↓
                                    assemble_module()
                                              ↓
                                    validate_bundle()  →  predictions-ready .py
```

### JSON schemas (`schemas/`)

| File | Purpose |
|------|---------|
| `triton_kernel.json` | General Triton bundle for any operator |
| `triton_vector_add.json` | Stricter template for 1D float32 vector add |

Required bundle fields:

- `wrapper_name` — exact TritonBench function name (e.g. `div`)
- `imports` — must include `torch`, `triton`, `triton.language as tl`
- `kernel_name`, `kernel_code` — `@triton.jit` kernel (no imports inside)
- `launcher_code` — wrapper function only
- `metadata` — dtype, device, block_size, etc. (hints for the model, not vLLM constraints)

### Generate structured output

```bash
python3 test_generation.py \
  --prompt "Functional Description: element-wise division with broadcasting..." \
  --structured-json schemas/triton_kernel.json \
  --wrapper-name div \
  --wrapper-signature "div(input, other, *, rounding_mode=None, out=None) -> Tensor" \
  -o out.json \
  --validate \
  --module-output out.py
```

### Generate with automatic repair (recommended)

Validates locally after each attempt; on failure re-prompts with errors (grid tuple, `.data_ptr()`, etc.):

```bash
python3 test_generation.py \
  --prompt "Functional Description: element-wise division..." \
  --structured-json schemas/triton_kernel.json \
  --wrapper-name div \
  --wrapper-signature "div(input, other, *, rounding_mode=None, out=None) -> Tensor" \
  --repair-attempts 3 \
  -o out.json
```

Writes `out.json` and `out.py` when validation passes.

Non-streaming is used automatically for structured JSON (more reliable than streaming).

### Validate an existing bundle

```bash
python3 validate_bundle.py out.json \
  --wrapper-name div \
  --wrapper-signature "div(input, other, *, rounding_mode=None, out=None) -> Tensor" \
  --assemble
```

### View assembled code

```bash
python3 show_out.py out.json --assemble
```

### Semantic checks (`validate_bundle`)

1. JSON Schema validation (`jsonschema`)
2. Assemble imports + kernel + launcher into one module
3. AST: `@triton.jit` on kernel, wrapper name and parameter names
4. Cross-reference: launcher calls `kernel_name`
5. Errors for int grid, missing `.data_ptr()`, missing `BLOCK_SIZE=` keyword
6. Rejects `tl.trunc` in kernel source
7. `compile()` syntax check

### Python API

```python
from triton_bundle import assemble_module, generate_structured, validate_bundle
from triton_bundle.prompts import structured_user_prompt
from triton_bundle.schema import load_schema

schema = load_schema("triton_kernel.json")
prompt = structured_user_prompt(
    instruction="...",
    wrapper_name="div",
    wrapper_signature="div(input, other, *, rounding_mode=None, out=None) -> Tensor",
)
bundle = generate_structured(
    endpoint="https://...modal.run",
    api_key="EMPTY",
    user_prompt=prompt,
    json_schema=schema,
)
result = validate_bundle(
    bundle,
    expected_wrapper_name="div",
    expected_signature="div(input, other, *, rounding_mode=None, out=None) -> Tensor",
)
module = result.module  # ready for deepbork predictions.jsonl
```

## Free-form generation

```bash
python3 test_generation.py --prompt "Write a Python function that checks if a number is prime."
```

## Structured outputs with curl

Use **non-streaming** for JSON payloads:

```bash
curl "$DEFAULT_ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Return JSON with kernel_name and block_size."}
    ],
    "structured_outputs": {
      "json": {
        "type": "object",
        "properties": {
          "kernel_name": {"type": "string"},
          "block_size": {"type": "integer"}
        },
        "required": ["kernel_name", "block_size"],
        "additionalProperties": false
      }
    }
  }'
```

## Cost and Shutdown

This app uses an **H100** GPU. Idle containers scale down after `scaledown_window=5 * MINUTES` in `vllm_inference.py`.

```bash
modal app stop example-vllm-inference
modal deploy vllm_inference.py   # redeploy after stop
```

## deepbork integration (next step)

deepbork expects `predictions.jsonl` with a full Python module in `predict`. Once validation passes locally:

```python
record = {"instruction": alpaca_instruction, "predict": result.module}
```

Wire this into `deepbork/main.py` by replacing `extract_code()` with `assemble_module()` when using structured generation.

## References

- [vLLM structured outputs](https://docs.vllm.ai/en/v0.19.1/features/structured_outputs/)
- [Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference)
