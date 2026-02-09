from __future__ import annotations

import math
import re

from .types import BackendProbe, ChatRequest


_BIG_MODEL_RE = re.compile(r"(?i)(?:^|[^0-9])(?:70b|72b|105b|180b|405b|671b)(?:$|[^0-9])")


def _model_in_probe(model: str, probe: BackendProbe) -> bool:
    m = model.strip()
    if not m:
        return False
    for info in probe.models:
        if info.id == m:
            return True
    return False


def score_backend_for_chat(req: ChatRequest, probe: BackendProbe) -> float:
    if not probe.ok:
        return -math.inf

    score = 0.0
    score += float(probe.backend.priority)

    # Kind preferences: OpenAI-compatible is the most generic surface.
    if probe.kind in ("openai", "exo"):
        score += 1.0
    if probe.kind == "ollama":
        score += 0.5

    # Requested model match is a strong signal.
    if req.model:
        if _model_in_probe(req.model, probe):
            score += 50.0
        else:
            # If the backend has an explicit model list and the model is absent, penalize.
            if probe.models:
                score -= 5.0

        # If model looks big, prefer exo (cluster capacity).
        if _BIG_MODEL_RE.search(req.model) and probe.kind == "exo":
            score += 10.0

    # Lower latency is slightly preferred (when available).
    if probe.latency_ms is not None:
        score += max(0.0, 5.0 - (probe.latency_ms / 200.0))

    return score

