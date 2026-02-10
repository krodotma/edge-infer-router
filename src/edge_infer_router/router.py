from __future__ import annotations

import math
import time
from typing import Any

from .exo import create_instance, get_instance_previews
from .http_util import HttpError, join_url, request_json
from .policy import score_backend_for_chat
from .probe import probe_backends
from .types import Backend, BackendProbe, ChatRequest, ChatMessage


def _split_backend_model(model: str) -> tuple[str | None, str]:
    # Allow explicit backend selection via "backend::model_id" to avoid ambiguity
    # across multiple providers. We use "::" (not ":") to avoid colliding with
    # Ollama tags like "llama3.2:latest".
    if "::" not in model:
        return None, model
    prefix, rest = model.split("::", 1)
    prefix = prefix.strip()
    rest = rest.strip()
    if not prefix or not rest:
        return None, model
    return prefix, rest


def _strip_backend_prefix(req: ChatRequest) -> tuple[str | None, ChatRequest]:
    if not req.model:
        return None, req
    backend_name, model_id = _split_backend_model(req.model)
    if backend_name is None:
        return None, req
    return backend_name, ChatRequest(
        messages=req.messages,
        model=model_id,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
        extra=req.extra,
    )


def choose_backend(backends: list[Backend], req: ChatRequest, *, timeout_s: float = 2.0) -> tuple[Backend, BackendProbe, list[BackendProbe]]:
    probes: list[BackendProbe] = probe_backends(backends, timeout_s=timeout_s)
    return choose_backend_from_probes(probes, req)


def choose_backend_from_probes(probes: list[BackendProbe], req: ChatRequest) -> tuple[Backend, BackendProbe, list[BackendProbe]]:
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
        _, _, data = request_json(
            method="POST",
            url=join_url(base_url, "/api/chat"),
            headers=backend_probe.backend.headers,
            payload=_ollama_chat_payload(req),
            timeout_s=timeout_s,
        )
        return _extract_ollama_text(data), data

    # Default: OpenAI-compatible chat completions
    _, _, data = request_json(
        method="POST",
        url=join_url(base_url, "/v1/chat/completions"),
        headers=backend_probe.backend.headers,
        payload=_openai_chat_payload(req),
        timeout_s=timeout_s,
    )
    return _extract_openai_text(data), data


def chat_with_selected_backend(
    backend: Backend,
    probe: BackendProbe,
    req: ChatRequest,
    *,
    probe_timeout_s: float = 2.0,
    chat_timeout_s: float = 60.0,
    exo_auto_instance: bool = False,
) -> tuple[str, Any]:
    try:
        return chat_once(probe, req, timeout_s=chat_timeout_s)
    except HttpError as e:
        # Best-effort: if this is exo and the model isn't deployed, try to create an instance and retry once.
        if exo_auto_instance and probe.kind == "exo" and req.model:
            if e.status in (400, 404) and (e.body_snippet or "").lower().find("model") != -1:
                previews = get_instance_previews(
                    backend.base_url,
                    headers=backend.headers,
                    model_id=req.model,
                    timeout_s=probe_timeout_s + 1.0,
                )
                if previews:
                    selected = None
                    for p in previews:
                        if p.get("model_id") == req.model:
                            selected = p
                            break
                    selected = selected or previews[0]
                    inst = selected.get("instance")
                    if isinstance(inst, dict):
                        create_instance(backend.base_url, inst, headers=backend.headers, timeout_s=probe_timeout_s + 1.0)
                        # Give exo a brief moment to register the instance before retrying.
                        time.sleep(0.05)
                        return chat_once(probe, req, timeout_s=chat_timeout_s)
        raise


def chat_with_routing(
    backends: list[Backend],
    req: ChatRequest,
    *,
    probe_timeout_s: float = 2.0,
    chat_timeout_s: float = 60.0,
    exo_auto_instance: bool = False,
) -> tuple[Backend, str, Any, list[BackendProbe]]:
    explicit_backend, normalized_req = _strip_backend_prefix(req)
    effective_backends = backends
    if explicit_backend is not None:
        effective_backends = [b for b in backends if b.name == explicit_backend]
        if not effective_backends:
            raise ValueError(f"unknown backend '{explicit_backend}' (from model selector)")

    backend, probe, probes = choose_backend(effective_backends, normalized_req, timeout_s=probe_timeout_s)

    text, raw = chat_with_selected_backend(
        backend,
        probe,
        normalized_req,
        probe_timeout_s=probe_timeout_s,
        chat_timeout_s=chat_timeout_s,
        exo_auto_instance=exo_auto_instance,
    )
    return backend, text, raw, probes


def simple_user_request(prompt: str, *, model: str | None = None) -> ChatRequest:
    msgs = [ChatMessage(role="user", content=prompt)]
    return ChatRequest(messages=msgs, model=model)
