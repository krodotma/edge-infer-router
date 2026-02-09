from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edge_infer_router.http_util import urlopen as real_urlopen
from edge_infer_router.router import chat_with_routing, simple_user_request
from edge_infer_router.types import Backend


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


class RouterTests(unittest.TestCase):
    def tearDown(self) -> None:
        from edge_infer_router import http_util

        http_util.urlopen = real_urlopen

    def test_routes_to_matching_model_backend(self):
        from edge_infer_router import http_util

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            if url == "http://a/v1/models":
                return FakeResp(200, {"data": [{"id": "foo"}]})
            if url == "http://b/v1/models":
                return FakeResp(200, {"data": [{"id": "bar"}]})
            if url.endswith("/v1/chat/completions"):
                return FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})
            raise make_http_error(url, 404, {"error": "unknown"})

        http_util.urlopen = fake_urlopen
        backends = [Backend(name="a", base_url="http://a"), Backend(name="b", base_url="http://b")]
        req = simple_user_request("hi", model="bar")
        backend, text, _raw, _probes = chat_with_routing(backends, req)
        self.assertEqual(backend.name, "b")
        self.assertEqual(text, "ok")

    def test_exo_auto_instance_create_and_retry(self):
        from edge_infer_router import http_util

        calls: list[str] = []

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            calls.append(url)

            if url.endswith("/v1/models"):
                # Exo can be OpenAI-compatible; probe will then check /state to confirm.
                return FakeResp(200, {"data": [{"id": "llama3.1-70b"}]})
            if url.endswith("/state"):
                return FakeResp(200, {"instances": []})
            if url.endswith("/v1/chat/completions"):
                # First attempt fails with model not found.
                if calls.count(url) == 1:
                    raise make_http_error(url, 400, {"error": "model not found"})
                return FakeResp(200, {"choices": [{"message": {"content": "after-up"}}]})
            if url.endswith("/instance/previews?model_id=llama3.1-70b"):
                return FakeResp(200, {"previews": [{"model_id": "llama3.1-70b", "instance": {"model_id": "llama3.1-70b"}}]})
            if url.endswith("/instance") and req.method == "POST":
                return FakeResp(200, {"message": "Command received.", "command_id": "cmd1"})

            raise make_http_error(url, 404, {"error": "unknown"})

        http_util.urlopen = fake_urlopen
        backends = [Backend(name="exo", base_url="http://exo")]
        req = simple_user_request("hi", model="llama3.1-70b")
        backend, text, _raw, _probes = chat_with_routing(backends, req, exo_auto_instance=True)
        self.assertEqual(backend.name, "exo")
        self.assertEqual(text, "after-up")
