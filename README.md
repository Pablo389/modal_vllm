# Modal vLLM Inference

Minimal repo for serving an open-source model with vLLM on Modal through an OpenAI-compatible API.

This example serves `google/gemma-4-26B-A4B-it` on an H200 GPU using Modal Volumes for Hugging Face and vLLM caches.

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

If you want to call OpenAI's hosted API with the same test client instead, use
OpenAI's official base URL and your real OpenAI API key:

```bash
DEFAULT_ENDPOINT=https://api.openai.com
OPENAI_API_KEY=sk-...
```

Authenticate Modal:

```bash
modal setup
```

If the Hugging Face model requires access approval, make sure your Modal environment has the needed Hugging Face token configured.

## Test

Run the local entrypoint. Modal will start the remote vLLM server, run a health check, and send a sample chat request:

```bash
modal run vllm_inference.py
```

You can pass a custom prompt:

```bash
modal run vllm_inference.py --content "Explain attention in transformers."
```

## Deploy

Deploy the OpenAI-compatible API:

```bash
modal deploy vllm_inference.py
```

After deployment, Modal prints a URL similar to:

```text
https://your-workspace-name--example-vllm-inference-serve.modal.run
```

Useful routes:

- `/health` checks server health.
- `/docs` opens the Swagger UI.
- `/v1/chat/completions` accepts OpenAI-compatible chat completion requests.

## Test the Deployed Endpoint

Load the endpoint from `.env` into your current shell:

```bash
set -a
source .env
set +a
```

First, verify the deployed server is healthy:

```bash
curl "$DEFAULT_ENDPOINT/health"
```

Then send a streaming chat completion request:

```bash
curl -N "$DEFAULT_ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Write a tiny Python function that adds two numbers."
      }
    ]
  }'
```

### Structured outputs with vLLM + XGrammar

This server can enforce output structure at decode time using vLLM
`structured_outputs` request fields (`json`, `regex`, `choice`, `grammar`).

JSON schemas in `schemas/`:

- `triton_kernel.json` — general Triton bundle (any op; flexible `constraints.dtype` and `constraints.extra`)
- `triton_vector_add.json` — stricter template for 1D float32 vector add (`uses_mask`, fixed dtype)

Structured JSON schema with `curl`:

```bash
curl -N "$DEFAULT_ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Return a JSON object with fields kernel_name (string) and block_size (integer)."
      }
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

You can also test structured outputs with the Python client:

```bash
python3 test_generation.py \
  --prompt "Return JSON with fields kernel_name and block_size." \
  --structured-json '{"type":"object","properties":{"kernel_name":{"type":"string"},"block_size":{"type":"integer"}},"required":["kernel_name","block_size"],"additionalProperties":false}'
```

Save clean, pretty-printed JSON to a file (validates JSON before writing):

```bash
python3 test_generation.py \
  --prompt "Generate a Triton vector-add for two 1D float32 CUDA tensors. Return only JSON matching the schema." \
  --structured-json schemas/triton_vector_add.json \
  -o out.json
```

View generated code from `out.json`:

```bash
python3 show_out.py out.json
```

General schema (softmax, matmul, etc. — describe the op in `--prompt`):

```bash
python3 test_generation.py \
  --prompt "Generate a Triton row-wise softmax for a 2D float32 CUDA tensor. Return only JSON matching the schema. Put axis and shapes in constraints.extra." \
  --structured-json schemas/triton_kernel.json \
  -o out.json
```

Or redirect stdout (`Model response:` is omitted automatically when stdout is not a TTY):

```bash
python3 test_generation.py --prompt "..." --structured-json schemas/triton_vector_add.json > out.json
```

You can also use the included Python client. It calls the API through the
official `openai` client library, so the deployed vLLM server is treated like
any other OpenAI-compatible chat completions endpoint:

```bash
python3 test_generation.py
```

The Python client reads `DEFAULT_ENDPOINT` and `OPENAI_API_KEY` from `.env`.
You can still override the endpoint from the command line:

```bash
python3 test_generation.py --endpoint "https://your-workspace-name--example-vllm-inference-serve.modal.run"
```

The Modal vLLM endpoint exposes the model as `llm`, which is the default model
used by `test_generation.py`.

With a custom prompt:

```bash
python3 test_generation.py --prompt "Write a Python function that checks if a number is prime."
```

To call OpenAI's hosted API instead of the Modal vLLM endpoint, pass the
official OpenAI base URL and an OpenAI model name:

```bash
python3 test_generation.py \
  --endpoint "https://api.openai.com" \
  --model "gpt-4.1-mini" \
  --prompt "Write a tiny Python function that adds two numbers."
```

## Cost and Shutdown

This app uses an H200 GPU, so idle time can get expensive. There are two separate things to manage:

- The deployed Modal App.
- The running GPU containers behind the `serve` Function.

According to Modal's docs, a deployed App persists until you stop it from the web UI or with `modal app stop`. However, Modal Functions scale independently, and by default they scale to zero when there are no live inputs, so a deployed App does not necessarily mean a GPU is always running.

The idle scale-down behavior is configured in `vllm_inference.py` on the `@app.function` decorator:

```python
@app.function(
    ...
    scaledown_window=2 * MINUTES,
    ...
)
```

This means Modal can keep an idle GPU container warm for up to 2 minutes after traffic stops. Lower values reduce idle GPU cost but make cold starts more common. Higher values reduce cold starts but keep the H200 reserved longer while idle. Modal documents the allowed `scaledown_window` range as 2 seconds to 20 minutes.

To see deployed or recently stopped apps:

```bash
modal app list
```

To stop this deployed app and terminate its running containers:

```bash
modal app stop example-vllm-inference
```

To skip the confirmation prompt:

```bash
modal app stop example-vllm-inference --yes
```

Stopping an app is destructive in Modal's terminology: you cannot restart that exact stopped deployment. To bring it back, deploy it again from this repo:

```bash
modal deploy vllm_inference.py
```

You can also stop the app from the Modal dashboard by opening the app overview page and using the red "Stop app" button.

## Notes

- vLLM and Transformers are installed in the Modal container image, not in the local Python environment.
- Local dependencies are only for running Modal and the test client.
- Set `FAST_BOOT = True` in `vllm_inference.py` if you prefer faster cold starts while iterating.
- Avoid setting `min_containers` for this Function unless you intentionally want at least one GPU container kept warm at all times.

## References

- [Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference)
- [How to deploy vLLM on Modal](https://modal.com/blog/how-to-deploy-vllm)
- [Modal Apps, Functions, and entrypoints](https://modal.com/docs/guide/apps)
- [Modal scaling and autoscaling](https://modal.com/docs/guide/scale)
- [Modal cold start performance](https://modal.com/docs/guide/cold-start)
- [Modal app CLI reference](https://modal.com/docs/reference/cli/app)
