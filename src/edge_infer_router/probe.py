from __future__ import annotations

import time
from typing import Any

from .http_util import HttpError, join_url, request_json
from .types import Backend, BackendKind, BackendProbe, ModelInfo


def _parse_openai_models(payload: Any) -> list[ModelInfo]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        out: list[ModelInfo] = []
        for item in payload["data"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                out.append(ModelInfo(id=item["id"], raw=item))
        return out
    return []


def _parse_exo_models(payload: Any) -> list[ModelInfo]:
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        out: list[ModelInfo] = []
        for item in payload["models"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                out.append(ModelInfo(id=item["id"], raw=item))
        return out
    return []


def _parse_ollama_tags(payload: Any) -> list[ModelInfo]:
    # Ollama: {"models": [{"name": "llama3.2:latest", ...}, ...]}
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        out: list[ModelInfo] = []
        for item in payload["models"]:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                out.append(ModelInfo(id=item["name"], raw=item))
        return out
    return []


def probe_backend(backend: Backend, *, timeout_s: float = 2.0) -> BackendProbe:
    t0 = time.perf_counter()
    details: dict[str, Any] = {}

    # 1) OpenAI models endpoint
    try:
        _, _, data = request_json(method="GET", url=join_url(backend.base_url, "/v1/models"), timeout_s=timeout_s)
        models = _parse_openai_models(data)
        details["openai_models"] = True

        # Distinguish exo vs generic OpenAI-compatible by probing an exo-specific endpoint.
        kind: BackendKind = "openai"
        try:
            _, _, state = request_json(method="GET", url=join_url(backend.base_url, "/state"), timeout_s=timeout_s)
            if isinstance(state, dict):
                kind = "exo"
                details["exo_state"] = True
        except HttpError:
            details["exo_state"] = False

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return BackendProbe(backend=backend, ok=True, kind=backend.kind_hint or kind, models=models, latency_ms=dt_ms, details=details)
    except HttpError as e:
        details["openai_models"] = False
        details["openai_models_error"] = str(e)

    # 2) exo models endpoint
    try:
        _, _, data = request_json(method="GET", url=join_url(backend.base_url, "/models"), timeout_s=timeout_s)
        models = _parse_exo_models(data)
        details["exo_models"] = True
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return BackendProbe(backend=backend, ok=True, kind=backend.kind_hint or "exo", models=models, latency_ms=dt_ms, details=details)
    except HttpError as e:
        details["exo_models"] = False
        details["exo_models_error"] = str(e)

    # 3) Ollama tags endpoint
    try:
        _, _, data = request_json(method="GET", url=join_url(backend.base_url, "/api/tags"), timeout_s=timeout_s)
        models = _parse_ollama_tags(data)
        details["ollama_tags"] = True
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return BackendProbe(backend=backend, ok=True, kind=backend.kind_hint or "ollama", models=models, latency_ms=dt_ms, details=details)
    except HttpError as e:
        details["ollama_tags"] = False
        details["ollama_tags_error"] = str(e)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return BackendProbe(backend=backend, ok=False, kind=backend.kind_hint or "unknown", models=[], latency_ms=dt_ms, details=details, error="unreachable or unknown API")

