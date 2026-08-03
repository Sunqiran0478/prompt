import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from promptos.adapters import OpenAICompatibleModel
from promptos.cli import _task_response_options


class _HTTPResponse:
    def __init__(self, content: str):
        self.payload = json.dumps({
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class OpenAICompatibleModelTests(unittest.TestCase):
    def test_json_mode_sends_response_format_and_max_tokens(self):
        model = OpenAICompatibleModel(
            "deepseek-chat",
            response_format="json_object",
            max_tokens=321,
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False),
            patch("urllib.request.urlopen", return_value=_HTTPResponse('{"ok": true}')) as urlopen,
        ):
            response = model._chat("Return JSON only.", '{"input": "x"}')

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 321)
        self.assertEqual(response.text, '{"ok": true}')

    def test_json_mode_retries_empty_or_invalid_content(self):
        model = OpenAICompatibleModel(
            "deepseek-chat",
            response_format="json_object",
            max_retries=1,
        )
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False),
            patch(
                "urllib.request.urlopen",
                side_effect=[_HTTPResponse(""), _HTTPResponse('{"ok": true}')],
            ) as urlopen,
            patch("time.sleep"),
        ):
            value = model.complete_json("Return JSON only.", "{}")

        self.assertEqual(value, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_response_format_is_part_of_cache_identity(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            plain = OpenAICompatibleModel("deepseek-chat", cache_dir=cache_dir)
            structured = OpenAICompatibleModel(
                "deepseek-chat",
                cache_dir=cache_dir,
                response_format="json_object",
            )
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False),
                patch(
                    "urllib.request.urlopen",
                    side_effect=[
                        _HTTPResponse('{"mode": "plain"}'),
                        _HTTPResponse('{"mode": "structured"}'),
                    ],
                ) as urlopen,
            ):
                plain._chat("JSON output.", "{}")
                structured._chat("JSON output.", "{}")
                cached = structured._chat("JSON output.", "{}")

            self.assertEqual(urlopen.call_count, 2)
            self.assertEqual(cached.text, '{"mode": "structured"}')
            self.assertTrue(cached.raw["_promptos_cached"])

    def test_invalid_structured_options_are_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleModel("deepseek-chat", response_format="json_schema")
        with self.assertRaises(ValueError):
            OpenAICompatibleModel("deepseek-chat", max_tokens=0)

    def test_finance_plugin_defaults_to_json_object(self):
        self.assertEqual(
            _task_response_options(Namespace(plugin="finance_classification")),
            ("json_object", 2048),
        )
        self.assertEqual(_task_response_options(Namespace(plugin=None)), (None, None))


if __name__ == "__main__":
    unittest.main()
