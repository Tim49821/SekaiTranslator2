import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.context.errors import is_context_length_error, provider_error_message
from modules.context.token_usage import (
    fallback_token_count,
    format_completion_token_usage,
    format_token_usage,
    messages_token_count,
)


class LLMTokenUsageTest(unittest.TestCase):
    def test_fallback_is_deterministic_for_ascii_and_cjk(self):
        self.assertEqual(fallback_token_count("abcdefgh你"), 3)

    def test_unknown_model_uses_fallback_plus_message_overhead(self):
        with patch("modules.context.token_usage._token_encoding_for_model", return_value=None):
            self.assertEqual(
                messages_token_count([{"role": "user", "content": "abcdefgh你"}], "unknown"),
                7,
            )

    def test_usage_formats_openai_and_cache_fields(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 8},
        }
        self.assertEqual(
            format_token_usage(usage),
            "prompt=10, completion=2, total=12, cache_hit=8",
        )
        self.assertEqual(
            format_completion_token_usage(SimpleNamespace(usage=usage)),
            "prompt=10, completion=2, total=12, cache_hit=8",
        )


class LLMContextErrorTest(unittest.TestCase):
    def test_recognizes_context_codes_and_messages(self):
        coded = SimpleNamespace(
            code="context_length_exceeded",
            status_code=400,
            body={},
        )
        self.assertTrue(is_context_length_error(coded))
        self.assertTrue(is_context_length_error(RuntimeError("maximum context length exceeded")))

    def test_rejects_unrelated_bad_requests_and_non_input_statuses(self):
        self.assertFalse(is_context_length_error(RuntimeError("max_tokens is invalid")))
        self.assertFalse(is_context_length_error(SimpleNamespace(
            code="context_length_exceeded",
            status_code=500,
            body={},
        )))

    def test_nested_provider_message_is_extracted(self):
        response = SimpleNamespace(
            json=lambda: {"error": {"message": "prompt is too long"}},
            text="",
        )
        self.assertEqual(provider_error_message(SimpleNamespace(response=response)), "prompt is too long")
