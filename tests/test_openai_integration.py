"""Opt-in test: never runs successfully without explicit integration secrets."""
from __future__ import annotations

import os
import unittest

from promptos.adapters import OpenAICompatibleModel


@unittest.skipUnless(
    os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL") and os.environ.get("PROMPTOS_INTEGRATION_MODEL"),
    "requires explicit OpenAI-compatible integration environment",
)
class OpenAICompatibleIntegrationTests(unittest.TestCase):
    def test_one_small_request_returns_content_and_usage(self):
        model = OpenAICompatibleModel(os.environ["PROMPTOS_INTEGRATION_MODEL"])
        response = model.generate("Return exactly the word PASS.", {"probe": "integration"})
        self.assertTrue(response.text.strip())
        self.assertEqual(response.model, os.environ["PROMPTOS_INTEGRATION_MODEL"])


if __name__ == "__main__":
    unittest.main()
