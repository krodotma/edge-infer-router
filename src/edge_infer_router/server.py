from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .http_util import HttpError
from .openai_compat import (
    chat_request_from_openai,
    openai_chat_from_text,
    openai_error,
    openai_models_from_probes,
    openai_usage_from_ollama,
)
from .probe_cache import ProbeCache
from .router import choose_backend_from_probes, chat_with_selected_backend
from .types import Backend, BackendProbe, ChatRequest


class RouterService:
    def __init__(
        self,
        backends: list[Backend],
        *,
        probe_timeout_s: float = 2.0,
        probe_cache_ttl_s: float = 3.0,
        chat_timeout_s: float = 60.0,
        exo_auto_instance: bool = False,
        gateway_token: str | None = None,
    ) -> None:
        self.backends = backends
        self.probe_timeout_s = float(probe_timeout_s)
        self.chat_timeout_s = float(chat_timeout_s)
        self.exo_auto_instance = bool(exo_auto_instance)
        self.gateway_token = (gateway_token or "").strip() or None
        self.cache = ProbeCache(ttl_s=float(probe_cache_ttl_s))
        self._lock = threading.Lock()

        # Side-effectful control-plane actions must not be exposed without auth.
        if self.exo_auto_instance and not self.gateway_token:
            raise ValueError("exo_auto_instance requires gateway auth (set --gateway-token or EIR_GATEWAY_TOKEN)")

    def get_probes(self) -> list[BackendProbe]:
        # Cache handles locking; we just keep it centralized for future policy.
        return self.cache.get_or_probe_all(self.backends, timeout_s=self.probe_timeout_s)

    def list_models_openai(self) -> list[dict[str, Any]]:
        probes = self.get_probes()
        return openai_models_from_probes(probes)

    def _strip_backend_prefix(self, req: ChatRequest) -> tuple[str | None, ChatRequest]:
        # Mirror router.py logic without importing private helpers.
        if not req.model or "::" not in req.model:
            return None, req
        prefix, rest = req.model.split("::", 1)
        prefix = prefix.strip()
        rest = rest.strip()
        if not prefix or not rest:
            return None, req
        return prefix, ChatRequest(
            messages=req.messages,
            model=rest,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=req.stream,
            extra=req.extra,
        )

    def route(self, req: ChatRequest) -> tuple[Backend, BackendProbe, ChatRequest, list[BackendProbe]]:
        explicit_backend, normalized_req = self._strip_backend_prefix(req)
        probes = self.get_probes()
        effective = probes
        if explicit_backend is not None:
            effective = [p for p in probes if p.backend.name == explicit_backend]
            if not effective:
                raise ValueError(f"unknown backend '{explicit_backend}' (from model selector)")
        backend, chosen, _all = choose_backend_from_probes(effective, normalized_req)
        return backend, chosen, normalized_req, probes

    def chat_openai(self, payload: Any) -> tuple[int, dict[str, Any]]:
        try:
            req = chat_request_from_openai(payload)
        except ValueError as e:
            return HTTPStatus.BAD_REQUEST, openai_error(str(e))

        if req.stream:
            return HTTPStatus.BAD_REQUEST, openai_error("stream=true is not supported by this gateway yet", code="stream_not_supported")

        requested_model = req.model
        try:
            backend, probe, normalized_req, probes = self.route(req)
        except Exception as e:
            return HTTPStatus.BAD_GATEWAY, openai_error(str(e), type_="routing_error")

        # Execute against chosen backend.
        try:
            text, raw = chat_with_selected_backend(
                backend,
                probe,
                normalized_req,
                probe_timeout_s=self.probe_timeout_s,
                chat_timeout_s=self.chat_timeout_s,
                exo_auto_instance=self.exo_auto_instance,
            )
        except HttpError as e:
            msg = f"{e}"
            if e.body_snippet:
                msg = f"{msg}: {e.body_snippet}"
            return HTTPStatus.BAD_GATEWAY, openai_error(msg, type_="upstream_error")

        # OpenAI-compatible backends can be returned as-is.
        if probe.kind in ("openai", "exo"):
            if isinstance(raw, dict):
                obj = raw
            else:
                obj = openai_chat_from_text(model=requested_model or normalized_req.model, text=str(raw), backend=backend.name)
            if requested_model and isinstance(obj, dict):
                obj = dict(obj)
                obj["model"] = requested_model
            return HTTPStatus.OK, obj

        # Ollama needs adaptation into OpenAI chat completion shape.
        usage = openai_usage_from_ollama(raw)
        obj = openai_chat_from_text(model=requested_model or normalized_req.model, text=text, backend=backend.name, usage=usage)
        if requested_model and isinstance(obj, dict):
            obj = dict(obj)
            obj["model"] = requested_model
        return HTTPStatus.OK, obj


def _read_json_body(handler: BaseHTTPRequestHandler, *, limit: int = 2 * 1024 * 1024) -> Any:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return None
    if length > limit:
        raise ValueError(f"request body too large ({length} bytes)")
    body = handler.rfile.read(length)
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"invalid JSON: {e}") from e


def _write_json(handler: BaseHTTPRequestHandler, status: int, obj: Any) -> None:
    data = json.dumps(obj).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(service: RouterService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Keep stdout/stderr quiet by default; the caller can wrap the process for logs.
            return

        def _authorized(self) -> bool:
            token = service.gateway_token
            if not token:
                return True
            got = (self.headers.get("Authorization") or "").strip()
            if got == f"Bearer {token}":
                return True
            _write_json(self, HTTPStatus.UNAUTHORIZED, openai_error("unauthorized", type_="authentication_error", code="unauthorized"))
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            path = urlparse(self.path).path
            if path in ("/healthz", "/health"):
                probes = service.get_probes()
                payload = {
                    "ok": True,
                    "ts": time.time(),
                    "backends": [
                        {
                            "name": p.backend.name,
                            "base_url": p.backend.base_url,
                            "kind": p.kind,
                            "ok": p.ok,
                            "models": len(p.models),
                            "latency_ms": p.latency_ms,
                        }
                        for p in probes
                    ],
                }
                _write_json(self, HTTPStatus.OK, payload)
                return

            if path == "/v1/models":
                models = service.list_models_openai()
                _write_json(self, HTTPStatus.OK, {"object": "list", "data": models})
                return

            _write_json(self, HTTPStatus.NOT_FOUND, openai_error("not found", type_="not_found"))

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            path = urlparse(self.path).path
            if path == "/v1/chat/completions":
                try:
                    payload = _read_json_body(self)
                except ValueError as e:
                    _write_json(self, HTTPStatus.BAD_REQUEST, openai_error(str(e)))
                    return

                status, obj = service.chat_openai(payload)
                _write_json(self, status, obj)
                return

            _write_json(self, HTTPStatus.NOT_FOUND, openai_error("not found", type_="not_found"))

    return Handler


def serve(
    backends: list[Backend],
    *,
    host: str = "127.0.0.1",
    port: int = 9333,
    probe_timeout_s: float = 2.0,
    probe_cache_ttl_s: float = 3.0,
    chat_timeout_s: float = 60.0,
    exo_auto_instance: bool = False,
    gateway_token: str | None = None,
) -> None:
    def _is_loopback_host(h: str) -> bool:
        s = (h or "").strip().lower()
        return s in ("127.0.0.1", "::1", "localhost")

    if not _is_loopback_host(host) and not ((gateway_token or "").strip()):
        raise ValueError("refusing to bind to a non-loopback host without gateway auth (--gateway-token or EIR_GATEWAY_TOKEN)")

    service = RouterService(
        backends,
        probe_timeout_s=probe_timeout_s,
        probe_cache_ttl_s=probe_cache_ttl_s,
        chat_timeout_s=chat_timeout_s,
        exo_auto_instance=exo_auto_instance,
        gateway_token=gateway_token,
    )
    handler = make_handler(service)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    httpd.serve_forever()
