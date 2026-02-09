from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from .probe import probe_backend
from .router import chat_with_routing, simple_user_request
from .types import Backend


def _parse_backends(values: list[str]) -> list[Backend]:
    out: list[Backend] = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"invalid backend '{v}'; expected NAME=URL")
        name, url = v.split("=", 1)
        out.append(Backend(name=name.strip(), base_url=url.strip()))
    return out


def _load_backends_from_env() -> list[str]:
    s = os.environ.get("EIR_BACKENDS", "").strip()
    if not s:
        return []
    # Comma-separated NAME=URL pairs.
    return [p.strip() for p in s.split(",") if p.strip()]


def cmd_detect(args: argparse.Namespace) -> int:
    backends = _parse_backends(args.backend or _load_backends_from_env())
    if not backends:
        print("No backends configured. Use --backend NAME=URL or EIR_BACKENDS.", file=sys.stderr)
        return 2

    probes = [probe_backend(b, timeout_s=args.timeout_s) for b in backends]
    if args.format == "json":
        print(json.dumps([asdict(p) for p in probes], indent=2, sort_keys=True))
    else:
        for p in probes:
            status = "ok" if p.ok else "fail"
            models = ", ".join(m.id for m in p.models[:5])
            if len(p.models) > 5:
                models += ", ..."
            print(f"{p.backend.name}: {status} kind={p.kind} models=[{models}]")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    backends = _parse_backends(args.backend or _load_backends_from_env())
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eir", description="Edge inference router (exo/vLLM/Ollama).")
    p.add_argument("--backend", action="append", default=[], help="Backend in NAME=URL form. Repeatable.")

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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

