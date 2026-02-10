from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edge_infer_router.openai_compat import chat_request_from_openai, openai_models_from_probes
from edge_infer_router.types import Backend, BackendProbe, ModelInfo


class OpenAICompatTests(unittest.TestCase):
    def test_chat_request_from_openai_extracts_extra(self):
        req = chat_request_from_openai(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.25,
                "max_tokens": 12,
                "stream": False,
                "top_p": 0.9,
            }
        )
        self.assertEqual(req.model, "m")
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, "user")
        self.assertEqual(req.messages[0].content, "hi")
        self.assertEqual(req.max_tokens, 12)
        self.assertIn("top_p", req.extra)

    def test_models_union_disambiguates_duplicates(self):
        a = BackendProbe(backend=Backend(name="a", base_url="http://a"), ok=True, kind="openai", models=[ModelInfo(id="m1"), ModelInfo(id="shared")])
        b = BackendProbe(backend=Backend(name="b", base_url="http://b"), ok=True, kind="openai", models=[ModelInfo(id="m2"), ModelInfo(id="shared")])
        models = openai_models_from_probes([a, b])
        ids = sorted(m["id"] for m in models)
        self.assertIn("m1", ids)
        self.assertIn("m2", ids)
        self.assertIn("a::shared", ids)
        self.assertIn("b::shared", ids)
        self.assertNotIn("shared", ids)

