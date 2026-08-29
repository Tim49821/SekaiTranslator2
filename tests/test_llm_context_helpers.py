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

    def test_usage_ignores_invalid_counts_and_only_derives_complete_totals(self):
        invalid = {
            "prompt_tokens": True,
            "completion_tokens": -1,
            "total_tokens": "12",
            "prompt_tokens_details": {
                "cached_tokens": False,
                "cache_miss_tokens": -2,
                "cache_write_tokens": "3",
            },
        }
        self.assertEqual(format_token_usage(invalid), "")
        self.assertEqual(format_token_usage({"prompt_tokens": 4}), "prompt=4")
        self.assertEqual(format_token_usage({"completion_tokens": 3}), "completion=3")
        self.assertEqual(
            format_token_usage({"prompt_tokens": 4, "completion_tokens": 3}),
            "prompt=4, completion=3, total=7",
        )

    def test_usage_formats_cache_miss_and_write_after_cache_hit(self):
        usage = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "cache_hit": 4,
            "cache_miss": 5,
            "cache_write": 6,
        }
        self.assertEqual(
            format_token_usage(usage),
            "prompt=1, completion=2, total=3, cache_hit=4, cache_miss=5, cache_write=6",
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

    def test_rejects_non_input_statuses_from_all_response_status_fields(self):
        for status_name in ("status", "http_status", "httpStatus"):
            with self.subTest(status_name=status_name):
                response = SimpleNamespace(
                    **{
                        status_name: 500,
                        "json": lambda: {"error": {"code": "context_length_exceeded"}},
                    }
                )
                self.assertFalse(is_context_length_error(SimpleNamespace(response=response)))
