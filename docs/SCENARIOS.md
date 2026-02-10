# Scenarios

This doc captures expected behavior for common edge setups.

## 1) Only one backend reachable

- `eir detect` shows exactly one backend `ok=true`
- `eir chat` routes to the only reachable backend

## 2) vLLM + Ollama both reachable

- If the requested `--model` exists on one backend, it is preferred.
- Otherwise, the policy prefers OpenAI-compatible `/v1/chat/completions` backends.

## 2.1) Force a Specific Backend

If you want to avoid ambiguity (same model name on multiple backends), you can force routing
by prefixing the model as `BACKEND::MODEL_ID` where `BACKEND` matches your configured backend name.

Example:

```bash
eir chat --model ollama::llama3.2:latest --prompt "..."
```

## 3) exo present for "big model" capacity

exo is typically interesting when:

- A single device cannot host the model (memory bound).
- You want to pool multiple consumer devices (edge cluster) into a single inference surface.

Policy behavior:

- If `--model` matches a model listed by exo, exo gets a strong boost.
- If `--model` looks "big" (e.g. contains `70b`, `72b`, `405b`, `671b`), exo gets an extra boost.

## 4) exo model not deployed yet

If `--exo-auto-instance` is enabled:

- On a model-not-found style error from chat, the router attempts:
  1. `GET /instance/previews?model_id=...`
  2. `POST /instance` with the preview's `instance` payload
  3. Retry the chat request once

If any step fails, the router returns the original error.

## 5) Per-Backend Auth Headers

If a backend requires auth, provide headers either via:

- `--header BACKEND:Header=Value` (repeatable), or
- `EIR_BACKEND_HEADERS_JSON` (backend name -> headers object)

Example:

```bash
export EIR_BACKEND_HEADERS_JSON='{"vllm":{"Authorization":"Bearer ..."}}'
eir detect
```

## 6) Gateway Server Mode

`eir serve` runs an OpenAI-compatible gateway:

- `GET /v1/models`
- `POST /v1/chat/completions`

It routes requests to the configured backends using the same policy.
