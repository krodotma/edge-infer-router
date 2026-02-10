from __future__ import annotations

import time
import uuid
from typing import Any

from .types import BackendProbe, ChatMessage, ChatRequest


def chat_request_from_openai(payload: Any) -> ChatRequest:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("model must be a string")

    messages_raw = payload.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        raise ValueError("messages must be a non-empty array")

    messages: list[ChatMessage] = []
    for i, m in enumerate(messages_raw):
        if not isinstance(m, dict):
            raise ValueError(f"messages[{i}] must be an object")
        role = m.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{i}].role must be a string")
        # content can be str or structured; keep as-is.
        if "content" not in m:
            raise ValueError(f"messages[{i}].content is required")
        msg_extra = {k: v for k, v in m.items() if k not in {"role", "content"}}
        messages.append(ChatMessage(role=role, content=m.get("content"), extra=msg_extra))

    temperature = payload.get("temperature")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))):
        raise ValueError("temperature must be a number")

    max_tokens = payload.get("max_tokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)):
        raise ValueError("max_tokens must be an integer")

    stream = payload.get("stream", False)
    if stream is not None and not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    stream = bool(stream)

    known = {"model", "messages", "temperature", "max_tokens", "stream"}
    extra = {k: v for k, v in payload.items() if k not in known}

    return ChatRequest(
        messages=messages,
        model=model,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
        stream=stream,
        extra=extra,
    )


def openai_error(
    message: str,
    *,
    code: str | None = None,
    type_: str = "invalid_request_error",
    param: str | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"message": message, "type": type_}
    if code:
        err["code"] = code
    if param:
        err["param"] = param
    return {"error": err}


def openai_models_from_probes(probes: list[BackendProbe]) -> list[dict[str, Any]]:
    # Strategy:
    # - If a model id is unique across backends, expose it as-is.
    # - If a model id appears on multiple backends, only expose explicit ids:
    #     backend::model_id
    model_to_backends: dict[str, set[str]] = {}
    for p in probes:
        if not p.ok:
            continue
        for m in p.models:
            model_to_backends.setdefault(m.id, set()).add(p.backend.name)

    out: list[dict[str, Any]] = []
    now = int(time.time())
    for model_id, backends in sorted(model_to_backends.items(), key=lambda kv: kv[0]):
        if len(backends) == 1:
            out.append({"id": model_id, "object": "model", "created": now, "owned_by": next(iter(backends))})
        else:
            for backend in sorted(backends):
                out.append(
                    {
                        "id": f"{backend}::{model_id}",
                        "object": "model",
                        "created": now,
                        "owned_by": backend,
                    }
                )
    return out


def openai_chat_from_text(
    *,
    model: str | None,
    text: str,
    backend: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    created = int(time.time())
    out_model = model or "unknown"
    # Keep the response OpenAI-compatible; avoid custom top-level fields.
    _ = backend  # reserved for future optional headers/metadata
    out: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": out_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    # Many OpenAI client SDKs expect `usage` to exist. If we don't have upstream
    # counts, report zeros rather than omitting the field.
    out["usage"] = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return out


def openai_usage_from_ollama(raw: Any) -> dict[str, int] | None:
    # Ollama may return token counts as:
    #   prompt_eval_count, eval_count
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_eval_count")
    completion = raw.get("eval_count")
    if isinstance(prompt, int) and isinstance(completion, int):
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    return None
