# edge-infer-router

An "edge inference router" that can:

- Probe multiple local inference backends (exo, OpenAI-compatible servers like vLLM, Ollama).
- Choose a backend based on a lightweight scoring policy.
- Send a chat completion request to the chosen backend.
- Optionally: if routed to exo and the model is not deployed, auto-create an exo instance from previews.

This repo is intentionally stdlib-only (no runtime dependencies).

## Install

```bash
python3 -m pip install -e .
```

## Configure Backends

Backends can be provided via repeated `--backend NAME=URL`, or `EIR_BACKENDS`:

```bash
export EIR_BACKENDS="exo=http://127.0.0.1:52415,vllm=http://127.0.0.1:8000,ollama=http://127.0.0.1:11434"
```

## Detect / Probe

```bash
eir detect
```

## Route + Chat

```bash
eir chat --model llama3.2:3b --prompt "Explain Kademlia in 5 sentences."
```

To force a specific backend (avoid ambiguity across providers), prefix the model with `BACKEND::`:

```bash
eir chat --model ollama::llama3.2:latest --prompt "..."
```

If routed to exo, you can allow "auto instance create" (best-effort):

```bash
eir chat --model llama3.1-70b --prompt "..." --exo-auto-instance
```

## Notes

- The router uses OpenAI Chat Completions (`/v1/chat/completions`) when available.
- Ollama is supported via the native `/api/chat` API when OpenAI endpoints are not present.
- exo management endpoints used (best-effort):
  - `GET /models`
  - `GET /state`
  - `GET /instance/previews?model_id=...`
  - `POST /instance` with `{"instance": ...}`

More routing examples in `docs/SCENARIOS.md`.
