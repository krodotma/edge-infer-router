from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


urlopen = urllib.request.urlopen


def join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    p = path if path.startswith("/") else f"/{path}"
    return f"{base}{p}"


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpError(Exception):
    def __init__(
        self,
        *,
        method: str,
        url: str,
        status: int | None,
        reason: str | None,
        body_snippet: str | None = None,
    ) -> None:
        super().__init__(f"{method} {url} -> {status} {reason or ''}".strip())
        self.method = method
        self.url = url
        self.status = status
        self.reason = reason
        self.body_snippet = body_snippet


def _read_body_snippet(body: bytes, limit: int = 2048) -> str:
    try:
        s = body[:limit].decode("utf-8", errors="replace")
    except Exception:
        return "<non-text body>"
    s = s.replace("\r", "").strip()
    return s


def request_bytes(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str] | None = None,
    payload: bytes | None = None,
    timeout_s: float = 5.0,
) -> HttpResponse:
    req = urllib.request.Request(url, data=payload, method=method.upper())
    for k, v in (headers or {}).items():
        req.add_header(k, v)

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            status = getattr(resp, "status", None) or resp.getcode()
            hdrs = getattr(resp, "headers", {}) or {}
            return HttpResponse(status=int(status), headers=dict(hdrs), body=body)
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read()  # type: ignore[assignment]
        except Exception:
            body = b""
        raise HttpError(
            method=method.upper(),
            url=url,
            status=getattr(e, "code", None),
            reason=getattr(e, "reason", None),
            body_snippet=_read_body_snippet(body),
        ) from e
    except (urllib.error.URLError, socket.timeout) as e:
        raise HttpError(method=method.upper(), url=url, status=None, reason=str(e)) from e


def request_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str] | None = None,
    payload: Any | None = None,
    timeout_s: float = 5.0,
) -> tuple[int, Mapping[str, str], Any]:
    body = None
    hdrs = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")

    resp = request_bytes(method=method, url=url, headers=hdrs, payload=body, timeout_s=timeout_s)
    if not resp.body:
        return resp.status, resp.headers, None
    try:
        data = json.loads(resp.body.decode("utf-8"))
    except Exception as e:
        raise HttpError(
            method=method.upper(),
            url=url,
            status=resp.status,
            reason=f"invalid json: {e}",
            body_snippet=_read_body_snippet(resp.body),
        ) from e
    return resp.status, resp.headers, data

