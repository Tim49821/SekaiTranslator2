import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

from modules.context.errors import is_context_length_error, provider_error_message
from modules.context.token_usage import (
    fallback_token_count,
    format_completion_token_usage,
    format_token_usage,
    messages_token_count,
)
from modules.context.history import (
    ContextAction,
    ContextReason,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    RenderedHistoryPage,
    RequestContext,
    eligible_history_for_request,
    recover_context_length,
    window_rebuild_reason,
)


def rendered(page_key, tokens):
    snapshot = HistoryPage(page_key, (page_key,), (page_key + "-translated",))
    return RenderedHistoryPage(snapshot, (("user", page_key), ("assistant", "done")), tokens)


class FakeProject:
    def __init__(self, keys):
        self.load_identity = object()
        self.pages = OrderedDict((key, []) for key in keys)


class LLMHistoryWindowTest(unittest.TestCase):
    def test_rebuild_selects_newest_suffix_below_low_water(self):
        project = FakeProject(["001.png", "002.png", "003.png", "004.png"])
        pages = {key: HistoryPage(key, (key,), ("done",)) for key in project.pages}
        history, diagnostic = eligible_history_for_request(
            window=None,
            project=project,
            page_key="004.png",
            previous_page=None,
            token_budget=10,
            rebuild_reason=ContextReason.WINDOW_EMPTY,
            snapshot_page=pages.get,
            render_page=lambda page: rendered(page.page_key, 3),
        )
        self.assertEqual([page.page_key for page in history], ["002.png", "003.png"])
        self.assertEqual(diagnostic.action, ContextAction.REBUILD)
        self.assertEqual(diagnostic.token_count, 6)

    def test_adjacent_request_grows_then_bulk_evicts(self):
        project = FakeProject(["001.png", "002.png", "003.png"])
        key = HistoryWindowKey(project.load_identity, (("model", "demo"),))
        first = rendered("001.png", 4)
        window = HistoryWindow(key, "001.png", (first,), 4)
        previous = HistoryPage("002.png", ("two",), ("둘",))

        history, diagnostic = eligible_history_for_request(
            window=window,
            project=project,
            page_key="003.png",
            previous_page=previous,
            token_budget=6,
            rebuild_reason=None,
            snapshot_page=lambda _key: None,
            render_page=lambda _page: rendered("002.png", 4),
        )
        self.assertEqual([page.page_key for page in history], ["002.png"])
        self.assertEqual(diagnostic.action, ContextAction.EVICT)
        self.assertEqual(diagnostic.evicted, 1)

    def test_context_recovery_removes_at_least_one_whole_page(self):
        request = RequestContext(
            history=(rendered("001.png", 3), rendered("002.png", 3)),
            history_budget=10,
            request_page_key="003.png",
        )
        recovered = recover_context_length(request)
        self.assertEqual([page.page_key for page in recovered.history], ["002.png"])
        self.assertEqual(recovered.diagnostic.action, ContextAction.CONTEXT_RECOVERY)

    def test_project_identity_and_nonadjacent_page_force_rebuild(self):
        project = FakeProject(["001.png", "002.png", "003.png"])
        key = HistoryWindowKey(project.load_identity, (("model", "demo"),))
        window = HistoryWindow(key, "001.png", (), 0)
        self.assertIsNone(window_rebuild_reason(window, project, "002.png", key))
        self.assertEqual(
            window_rebuild_reason(window, project, "003.png", key),
            ContextReason.NON_ADJACENT,
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
