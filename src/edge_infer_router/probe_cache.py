from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .probe import probe_backends
from .types import Backend, BackendProbe


@dataclass(frozen=True)
class BackendKey:
    name: str
    base_url: str


@dataclass(frozen=True)
class ProbeCacheEntry:
    probe: BackendProbe
    expires_ts: float


class ProbeCache:
    def __init__(self, *, ttl_s: float = 3.0) -> None:
        self.ttl_s = float(ttl_s)
        self._lock = threading.Lock()
        self._entries: dict[BackendKey, ProbeCacheEntry] = {}

    def _key(self, backend: Backend) -> BackendKey:
        return BackendKey(name=backend.name, base_url=backend.base_url)

    def get(self, backend: Backend) -> BackendProbe | None:
        key = self._key(backend)
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry and entry.expires_ts > now:
                return entry.probe
        return None

    def set(self, backend: Backend, probe: BackendProbe) -> None:
        key = self._key(backend)
        expires = time.time() + self.ttl_s
        with self._lock:
            self._entries[key] = ProbeCacheEntry(probe=probe, expires_ts=expires)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get_or_probe_all(self, backends: list[Backend], *, timeout_s: float = 2.0) -> list[BackendProbe]:
        cached: dict[int, BackendProbe] = {}
        missing: list[tuple[int, Backend]] = []
        for i, b in enumerate(backends):
            p = self.get(b)
            if p is None:
                missing.append((i, b))
            else:
                cached[i] = p

        if missing:
            new_backends = [b for _i, b in missing]
            new_probes = probe_backends(new_backends, timeout_s=timeout_s)
            for (i, b), p in zip(missing, new_probes):
                self.set(b, p)
                cached[i] = p

        return [cached[i] for i in range(len(backends))]

