from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
        _, _, data = request_json(
            method="GET",
            url=join_url(backend.base_url, "/v1/models"),
            headers=backend.headers,
            timeout_s=timeout_s,
        )
        models = _parse_openai_models(data)
        details["openai_models"] = True

        # Distinguish exo vs generic OpenAI-compatible by probing an exo-specific endpoint.
        kind: BackendKind = "openai"
        try:
            _, _, state = request_json(
                method="GET",
                url=join_url(backend.base_url, "/state"),
                headers=backend.headers,
                timeout_s=timeout_s,
            )
            # Heuristic: exo /state includes an "instances" array.
            if isinstance(state, dict) and isinstance(state.get("instances"), list):
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
        _, _, data = request_json(
            method="GET",
            url=join_url(backend.base_url, "/models"),
            headers=backend.headers,
            timeout_s=timeout_s,
        )
        models = _parse_exo_models(data)
        details["exo_models"] = True
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return BackendProbe(backend=backend, ok=True, kind=backend.kind_hint or "exo", models=models, latency_ms=dt_ms, details=details)
    except HttpError as e:
        details["exo_models"] = False
        details["exo_models_error"] = str(e)

    # 3) Ollama tags endpoint
    try:
        _, _, data = request_json(
            method="GET",
            url=join_url(backend.base_url, "/api/tags"),
            headers=backend.headers,
            timeout_s=timeout_s,
        )
        models = _parse_ollama_tags(data)
        details["ollama_tags"] = True
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return BackendProbe(backend=backend, ok=True, kind=backend.kind_hint or "ollama", models=models, latency_ms=dt_ms, details=details)
    except HttpError as e:
        details["ollama_tags"] = False
        details["ollama_tags_error"] = str(e)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return BackendProbe(backend=backend, ok=False, kind=backend.kind_hint or "unknown", models=[], latency_ms=dt_ms, details=details, error="unreachable or unknown API")


def probe_backends(
    backends: list[Backend],
    *,
    timeout_s: float = 2.0,
    max_workers: int | None = None,
) -> list[BackendProbe]:
    if not backends:
        return []
    if len(backends) == 1:
        return [probe_backend(backends[0], timeout_s=timeout_s)]

    workers = max_workers or min(16, len(backends))
    probes: dict[int, BackendProbe] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe_backend, b, timeout_s=timeout_s): i for i, b in enumerate(backends)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                probes[i] = fut.result()
            except Exception as e:
                # Defensive: probe_backend should return BackendProbe, not raise.
                b = backends[i]
                probes[i] = BackendProbe(
                    backend=b,
                    ok=False,
                    kind=b.kind_hint or "unknown",
                    models=[],
                    latency_ms=None,
                    details={"exception": str(e)},
                    error=str(e),
                )
    return [probes[i] for i in range(len(backends))]
