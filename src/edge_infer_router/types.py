from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


BackendKind = Literal["unknown", "openai", "exo", "ollama"]


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    kind_hint: BackendKind | None = None
    priority: int = 0
    headers: dict[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendProbe:
    backend: Backend
    ok: bool
    kind: BackendKind
    models: list[ModelInfo] = field(default_factory=list)
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: Any


@dataclass(frozen=True)
class ChatRequest:
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
