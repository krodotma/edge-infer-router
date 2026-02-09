from __future__ import annotations

from typing import Any

from .http_util import join_url, request_json


def get_state(base_url: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    _, _, data = request_json(method="GET", url=join_url(base_url, "/state"), timeout_s=timeout_s)
    if isinstance(data, dict):
        return data
    return {"raw": data}


def get_instance_previews(base_url: str, *, model_id: str | None = None, timeout_s: float = 3.0) -> list[dict[str, Any]]:
    path = "/instance/previews"
    if model_id:
        path += f"?model_id={model_id}"
    _, _, data = request_json(method="GET", url=join_url(base_url, path), timeout_s=timeout_s)
    if isinstance(data, dict) and isinstance(data.get("previews"), list):
        out: list[dict[str, Any]] = []
        for p in data["previews"]:
            if isinstance(p, dict):
                out.append(p)
        return out
    return []


def create_instance(base_url: str, instance: dict[str, Any], *, timeout_s: float = 3.0) -> dict[str, Any]:
    _, _, data = request_json(method="POST", url=join_url(base_url, "/instance"), payload={"instance": instance}, timeout_s=timeout_s)
    if isinstance(data, dict):
        return data
    return {"raw": data}


def delete_instance(base_url: str, instance_id: str, *, timeout_s: float = 3.0) -> dict[str, Any]:
    _, _, data = request_json(method="DELETE", url=join_url(base_url, f"/instance/{instance_id}"), timeout_s=timeout_s)
    if isinstance(data, dict):
        return data
    return {"raw": data}

