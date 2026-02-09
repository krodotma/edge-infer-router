from __future__ import annotations

import math
from typing import Any

from .exo import create_instance, get_instance_previews
from .http_util import HttpError, join_url, request_json
from .policy import score_backend_for_chat
from .probe import probe_backend
from .types import Backend, BackendProbe, ChatRequest, ChatMessage


def choose_backend(backends: list[Backend], req: ChatRequest, *, timeout_s: float = 2.0) -> tuple[Backend, BackendProbe, list[BackendProbe]]:
    probes: list[BackendProbe] = [probe_backend(b, timeout_s=timeout_s) for b in backends]
    best: tuple[float, int] | None = None
    for i, p in enumerate(probes):
        s = score_backend_for_chat(req, p)
        if best is None or s > best[0]:
            best = (s, i)

    if best is None or best[0] == -math.inf:
        raise RuntimeError("no reachable backends")

    chosen_probe = probes[best[1]]
    return chosen_probe.backend, chosen_probe, probes


def _openai_chat_payload(req: ChatRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "stream": bool(req.stream),
    }
    if req.model:
        payload["model"] = req.model
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    payload.update(req.extra or {})
    return payload


def _ollama_chat_payload(req: ChatRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "stream": bool(req.stream),
    }
    if req.model:
        payload["model"] = req.model
    payload.update(req.extra or {})
    return payload


def _extract_openai_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
    return str(data)


def _extract_ollama_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)
    msg = data.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    if isinstance(data.get("response"), str):
        return data["response"]
    return str(data)


def chat_once(backend_probe: BackendProbe, req: ChatRequest, *, timeout_s: float = 20.0) -> tuple[str, Any]:
    base_url = backend_probe.backend.base_url
    if backend_probe.kind == "ollama":
        _, _, data = request_json(method="POST", url=join_url(base_url, "/api/chat"), payload=_ollama_chat_payload(req), timeout_s=timeout_s)
        return _extract_ollama_text(data), data

    # Default: OpenAI-compatible chat completions
    _, _, data = request_json(method="POST", url=join_url(base_url, "/v1/chat/completions"), payload=_openai_chat_payload(req), timeout_s=timeout_s)
    return _extract_openai_text(data), data


def chat_with_routing(
    backends: list[Backend],
    req: ChatRequest,
    *,
    probe_timeout_s: float = 2.0,
    chat_timeout_s: float = 60.0,
    exo_auto_instance: bool = False,
) -> tuple[Backend, str, Any, list[BackendProbe]]:
    backend, probe, probes = choose_backend(backends, req, timeout_s=probe_timeout_s)

    try:
        text, raw = chat_once(probe, req, timeout_s=chat_timeout_s)
        return backend, text, raw, probes
    except HttpError as e:
        # Best-effort: if this is exo and the model isn't deployed, try to create an instance and retry once.
        if exo_auto_instance and probe.kind == "exo" and req.model:
            if e.status in (400, 404) and (e.body_snippet or "").lower().find("model") != -1:
                previews = get_instance_previews(backend.base_url, model_id=req.model, timeout_s=probe_timeout_s + 1.0)
                if previews:
                    inst = previews[0].get("instance")
                    if isinstance(inst, dict):
                        create_instance(backend.base_url, inst, timeout_s=probe_timeout_s + 1.0)
                        text, raw = chat_once(probe, req, timeout_s=chat_timeout_s)
                        return backend, text, raw, probes
        raise


def simple_user_request(prompt: str, *, model: str | None = None) -> ChatRequest:
    msgs = [ChatMessage(role="user", content=prompt)]
    return ChatRequest(messages=msgs, model=model)

