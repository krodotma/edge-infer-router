from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edge_infer_router.http_util import urlopen as real_urlopen
from edge_infer_router.types import Backend
from edge_infer_router.probe import probe_backend


class FakeResp(io.BytesIO):
    def __init__(self, status: int, payload: object):
        super().__init__(json.dumps(payload).encode("utf-8"))
        self.status = status
        self._headers = {"Content-Type": "application/json"}

    def getcode(self) -> int:
        return self.status

    @property
    def headers(self):
        return self._headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def make_http_error(url: str, code: int, payload: object) -> urllib.error.HTTPError:
    body = io.BytesIO(json.dumps(payload).encode("utf-8"))
    return urllib.error.HTTPError(url, code, "err", hdrs=None, fp=body)


class ProbeTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Ensure any accidental patching is undone.
        from edge_infer_router import http_util

        http_util.urlopen = real_urlopen

    def test_probe_exo_via_models(self):
        from edge_infer_router import http_util

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            if url.endswith("/v1/models"):
                raise make_http_error(url, 404, {"error": "nope"})
            if url.endswith("/models"):
                return FakeResp(200, {"models": [{"id": "llama3.1-70b"}]})
            raise make_http_error(url, 404, {"error": "unknown"})

        http_util.urlopen = fake_urlopen
        p = probe_backend(Backend(name="exo", base_url="http://x"))
        self.assertTrue(p.ok)
        self.assertEqual(p.kind, "exo")
        self.assertEqual([m.id for m in p.models], ["llama3.1-70b"])

    def test_probe_ollama_via_tags(self):
        from edge_infer_router import http_util

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            if url.endswith("/v1/models"):
                raise make_http_error(url, 404, {"error": "nope"})
            if url.endswith("/models"):
                raise make_http_error(url, 404, {"error": "nope"})
            if url.endswith("/api/tags"):
                return FakeResp(200, {"models": [{"name": "llama3.2:latest"}]})
            raise make_http_error(url, 404, {"error": "unknown"})

        http_util.urlopen = fake_urlopen
        p = probe_backend(Backend(name="ollama", base_url="http://x"))
        self.assertTrue(p.ok)
        self.assertEqual(p.kind, "ollama")
        self.assertEqual([m.id for m in p.models], ["llama3.2:latest"])
