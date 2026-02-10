from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edge_infer_router.http_util import urlopen as real_urlopen
from edge_infer_router.server import RouterService
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


class ServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        from edge_infer_router import http_util

        http_util.urlopen = real_urlopen

    def test_models_and_chat_openai_backend(self):
        from edge_infer_router import http_util

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            if url == "http://a/v1/models":
                return FakeResp(200, {"data": [{"id": "m1"}]})
            if url == "http://a/state":
                raise make_http_error(url, 404, {"error": "nope"})
            if url == "http://a/v1/chat/completions":
                return FakeResp(200, {"id": "x", "object": "chat.completion", "choices": [{"message": {"content": "ok"}}]})
            raise make_http_error(url, 404, {"error": "unknown"})

        http_util.urlopen = fake_urlopen
        service = RouterService([Backend(name="a", base_url="http://a")], probe_cache_ttl_s=60.0)

        models = service.list_models_openai()
        ids = [m["id"] for m in models]
        self.assertIn("m1", ids)

        status, obj = service.chat_openai({"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(int(status), 200)
        self.assertEqual(obj["choices"][0]["message"]["content"], "ok")

    def test_chat_ollama_backend_is_adapted(self):
        from edge_infer_router import http_util

        def fake_urlopen(req, timeout=0):
            url = req.full_url
            if url == "http://o/v1/models":
                raise make_http_error(url, 404, {"error": "nope"})
            if url == "http://o/models":
                raise make_http_error(url, 404, {"error": "nope"})
            if url == "http://o/api/tags":
                return FakeResp(200, {"models": [{"name": "llama3.2:latest"}]})
            if url == "http://o/api/chat":
                return FakeResp(200, {"message": {"content": "hello"}})
            raise make_http_error(url, 404, {"error": "unknown"})

        http_util.urlopen = fake_urlopen
        service = RouterService([Backend(name="ollama", base_url="http://o")], probe_cache_ttl_s=60.0)

        status, obj = service.chat_openai({"model": "llama3.2:latest", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(int(status), 200)
        self.assertEqual(obj["object"], "chat.completion")
        self.assertEqual(obj["choices"][0]["message"]["content"], "hello")

