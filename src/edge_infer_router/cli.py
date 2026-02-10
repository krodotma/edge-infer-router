from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from .probe import probe_backends
from .router import chat_with_routing, simple_user_request
from .server import serve
from .types import Backend


def _load_backend_headers_from_env() -> dict[str, dict[str, str]]:
    raw = os.environ.get("EIR_BACKEND_HEADERS_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"EIR_BACKEND_HEADERS_JSON is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("EIR_BACKEND_HEADERS_JSON must be a JSON object mapping backend name -> headers object")
    out: dict[str, dict[str, str]] = {}
    for backend_name, headers in data.items():
        if not isinstance(backend_name, str) or not isinstance(headers, dict):
            continue
        h: dict[str, str] = {}
        for k, v in headers.items():
            if isinstance(k, str) and isinstance(v, str):
                h[k] = v
        out[backend_name] = h
    return out


def _parse_backend_header_specs(values: list[str]) -> dict[str, dict[str, str]]:
    # Spec: BACKEND:Header=Value (repeatable)
    out: dict[str, dict[str, str]] = {}
    for v in values:
        if ":" not in v or "=" not in v:
            raise ValueError(f"invalid header '{v}'; expected BACKEND:Header=Value")
        backend, rest = v.split(":", 1)
        header, value = rest.split("=", 1)
        backend = backend.strip()
        header = header.strip()
        value = value.strip()
        if not backend or not header:
            raise ValueError(f"invalid header '{v}'; expected BACKEND:Header=Value")
        out.setdefault(backend, {})[header] = value
    return out


def _parse_backends(values: list[str], *, headers_by_backend: dict[str, dict[str, str]] | None = None) -> list[Backend]:
    out: list[Backend] = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"invalid backend '{v}'; expected NAME=URL")
        name, url = v.split("=", 1)
        name = name.strip()
        url = url.strip()
        headers = dict((headers_by_backend or {}).get(name, {}))
        out.append(Backend(name=name, base_url=url, headers=headers))
    return out


def _load_backends_from_env() -> list[str]:
    s = os.environ.get("EIR_BACKENDS", "").strip()
    if not s:
        return []
    # Comma-separated NAME=URL pairs.
    return [p.strip() for p in s.split(",") if p.strip()]


def _merge_headers(*maps: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for m in maps:
        for backend, hdrs in m.items():
            merged.setdefault(backend, {}).update(hdrs)
    return merged


def _build_backends(args: argparse.Namespace) -> list[Backend]:
    headers_by_backend = _merge_headers(_load_backend_headers_from_env(), _parse_backend_header_specs(args.header or []))
    backends = _parse_backends(args.backend or _load_backends_from_env(), headers_by_backend=headers_by_backend)
    return backends


def _probe_to_public_dict(p) -> dict:
    # Avoid leaking secrets in JSON output.
    return {
        "backend": {
            "name": p.backend.name,
            "base_url": p.backend.base_url,
            "kind_hint": p.backend.kind_hint,
            "priority": p.backend.priority,
            "header_keys": sorted(list((p.backend.headers or {}).keys())),
        },
        "ok": p.ok,
        "kind": p.kind,
        "models": [asdict(m) for m in p.models],
        "latency_ms": p.latency_ms,
        "details": p.details,
        "error": p.error,
    }


def cmd_detect(args: argparse.Namespace) -> int:
    backends = _build_backends(args)
    if not backends:
        print("No backends configured. Use --backend NAME=URL or EIR_BACKENDS.", file=sys.stderr)
        return 2

    probes = probe_backends(backends, timeout_s=args.timeout_s)
    if args.format == "json":
        print(json.dumps([_probe_to_public_dict(p) for p in probes], indent=2, sort_keys=True))
    else:
        for p in probes:
            status = "ok" if p.ok else "fail"
            models = ", ".join(m.id for m in p.models[:5])
            if len(p.models) > 5:
                models += ", ..."
            print(f"{p.backend.name}: {status} kind={p.kind} models=[{models}]")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    backends = _build_backends(args)
    if not backends:
        print("No backends configured. Use --backend NAME=URL or EIR_BACKENDS.", file=sys.stderr)
        return 2

    req = simple_user_request(args.prompt, model=args.model)
    backend, text, _raw, probes = chat_with_routing(
        backends,
        req,
        probe_timeout_s=args.probe_timeout_s,
        chat_timeout_s=args.chat_timeout_s,
        exo_auto_instance=args.exo_auto_instance,
    )
    if args.verbose:
        for p in probes:
            print(f"[probe] {p.backend.name}: ok={p.ok} kind={p.kind} models={len(p.models)} latency_ms={p.latency_ms}", file=sys.stderr)
        print(f"[route] chosen={backend.name} base_url={backend.base_url}", file=sys.stderr)
    print(text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    backends = _build_backends(args)
    if not backends:
        print("No backends configured. Use --backend NAME=URL or EIR_BACKENDS.", file=sys.stderr)
        return 2

    serve(
        backends,
        host=args.host,
        port=args.port,
        probe_timeout_s=args.probe_timeout_s,
        probe_cache_ttl_s=args.probe_cache_ttl_s,
        chat_timeout_s=args.chat_timeout_s,
        exo_auto_instance=args.exo_auto_instance,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eir", description="Edge inference router (exo/vLLM/Ollama).")
    p.add_argument("--backend", action="append", default=[], help="Backend in NAME=URL form. Repeatable.")
    p.add_argument("--header", action="append", default=[], help="Per-backend header in BACKEND:Header=Value form. Repeatable.")

    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="Probe configured backends.")
    d.add_argument("--timeout-s", type=float, default=2.0)
    d.add_argument("--format", choices=["text", "json"], default="text")
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser("chat", help="Route and send a chat request.")
    c.add_argument("--model", default=None)
    c.add_argument("--prompt", required=True)
    c.add_argument("--probe-timeout-s", type=float, default=2.0)
    c.add_argument("--chat-timeout-s", type=float, default=60.0)
    c.add_argument("--exo-auto-instance", action="store_true")
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_chat)

    s = sub.add_parser("serve", help="Run an OpenAI-compatible gateway server.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9333)
    s.add_argument("--probe-timeout-s", type=float, default=2.0)
    s.add_argument("--probe-cache-ttl-s", type=float, default=3.0)
    s.add_argument("--chat-timeout-s", type=float, default=60.0)
    s.add_argument("--exo-auto-instance", action="store_true")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
