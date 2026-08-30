"""Provider-independent adapter services for LLM translation context."""

import os

from utils.config import RunStatus

from .glossary import (
    GLOSSARY_MODE_ALL,
    GLOSSARY_MODE_MATCHING,
    GLOSSARY_MODES,
    load_glossary,
    select_glossary,
)
from .history import (
    ContextReason,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    RequestContext,
    eligible_history_for_request,
    window_rebuild_reason,
)
from .params import (
    CONTEXT_INVALIDATION_KEYS,
    CONTEXT_MODE_HISTORY,
    CONTEXT_MODE_PAGE,
    CONTEXT_MODES,
)


_DEFAULT_HISTORY_TOKEN_BUDGET = 4096


class LLMContextAdapterMixin:
    """Compose immutable history and glossary snapshots for one request."""

    def _context_model_name(self) -> str:
        raise NotImplementedError

    def _context_prompt_signature(self) -> str:
        raise NotImplementedError

    def _render_history_page(self, page: HistoryPage):
        raise NotImplementedError

    def _clear_pending_request_context(self):
        self._pending_request_context = None

    def _clear_history_window(self):
        self._history_window = None
        self._clear_pending_request_context()

    def _stage_request_context(
        self,
        request_context,
        commit_history_window,
    ):
        self._pending_request_context = (
            request_context
            if commit_history_window
            and isinstance(request_context, RequestContext)
            else None
        )

    def _finalize_translation(self, success):
        pending_context = getattr(
            self,
            "_pending_request_context",
            None,
        )
        self._clear_pending_request_context()
        if success:
            self._commit_request_context(pending_context)
        return super()._finalize_translation(success)

    def _commit_request_context(self, request_context):
        if (
            request_context is None
            or request_context.window_key is None
            or request_context.request_page_key is None
        ):
            return
        self._history_window = HistoryWindow(
            key=request_context.window_key,
            request_page_key=request_context.request_page_key,
            history=request_context.history,
            token_count=sum(page.token_count for page in request_context.history),
        )

    def _selected_glossary(self, request_context, sources):
        if request_context is None or not request_context.glossary:
            return ()
        if request_context.glossary_mode == GLOSSARY_MODE_ALL:
            return request_context.glossary
        return select_glossary(
            request_context.glossary,
            sources,
            request_context.glossary_mode,
        )

    def _snapshot_history_page(self, project, page_key):
        pages = getattr(project, "pages", None)
        image_info = getattr(project, "_image_info", None)
        try:
            blocks = pages[page_key]
            metadata = image_info[page_key]
            finish_code = metadata.get("finish_code", 0)
        except (AttributeError, KeyError, TypeError):
            return None

        try:
            is_translated = bool(finish_code & RunStatus.FIN_TRANSLATE)
        except TypeError:
            return None
        if not is_translated:
            return None

        if "translation_target" in metadata and str(
            metadata["translation_target"]
        ) != str(self.lang_target):
            return None

        sources = []
        translations = []
        try:
            for block in blocks:
                source = str(block.get_text() or "")
                if not source.strip():
                    continue
                translation = str(getattr(block, "translation", "") or "")
                if (
                    not translation.strip()
                    or translation.lstrip().startswith("[ERROR:")
                ):
                    return None
                sources.append(source)
                translations.append(translation)
        except (AttributeError, TypeError):
            return None

        if not sources:
            return None
        return HistoryPage(
            page_key=str(page_key),
            sources=tuple(sources),
            translations=tuple(translations),
        )

    def _snapshot_request_context(self, project, page_key):
        context_mode = self._runtime_context_mode()
        history_enabled = context_mode == CONTEXT_MODE_HISTORY
        glossary_path = self._runtime_glossary_path()
        glossary_enabled = bool(glossary_path)

        if not history_enabled:
            self._clear_history_window()
        if not history_enabled and not glossary_enabled:
            return None

        glossary_mode = self._runtime_glossary_mode()
        glossary = load_glossary(glossary_path) if glossary_enabled else ()
        request_page_key = self._runtime_page_key(page_key)

        if not history_enabled:
            return RequestContext(
                history=(),
                glossary=glossary,
                glossary_mode=glossary_mode,
                request_page_key=request_page_key,
            )

        history_budget = self._runtime_history_budget()
        window_key = HistoryWindowKey(
            load_identity=getattr(project, "load_identity", None),
            settings=(
                ("source_language", str(self.lang_source)),
                ("target_language", str(self.lang_target)),
                ("model", self._context_model_name()),
                ("prompt_signature", self._context_prompt_signature()),
                ("token_budget", history_budget),
            ),
        )
        rebuild_reason = window_rebuild_reason(
            getattr(self, "_history_window", None),
            project,
            request_page_key,
            window_key,
        )
        previous_page = self._previous_history_page(
            project,
            request_page_key,
        )
        if rebuild_reason is None and previous_page is None:
            rebuild_reason = ContextReason.PREVIOUS_INCOMPLETE

        history, diagnostic = eligible_history_for_request(
            window=getattr(self, "_history_window", None),
            project=project,
            page_key=request_page_key,
            previous_page=previous_page,
            token_budget=history_budget,
            rebuild_reason=rebuild_reason,
            snapshot_page=lambda candidate_key: self._snapshot_history_page(
                project, candidate_key
            ),
            render_page=self._render_history_page,
        )
        self.logger.debug(str(diagnostic))
        return RequestContext(
            history=history,
            glossary=glossary,
            glossary_mode=glossary_mode,
            history_budget=history_budget,
            window_key=window_key,
            request_page_key=request_page_key,
            diagnostic=diagnostic,
        )

    def _translate_with_context(
        self,
        src_list,
        *,
        project=None,
        page_key=None,
        commit_history_window=False,
    ):
        self._clear_pending_request_context()
        request_context = self._snapshot_request_context(project, page_key)
        return self._translate(
            src_list,
            request_context=request_context,
            page_key=page_key,
            commit_history_window=commit_history_window,
        )

    def updateParam(self, param_key: str, param_content):
        result = super().updateParam(param_key, param_content)
        if param_key in CONTEXT_INVALIDATION_KEYS:
            self._clear_history_window()
        return result

    def unload_model(self, empty_cache=False):
        result = super().unload_model(empty_cache=empty_cache)
        self._clear_history_window()
        return result

    def _previous_history_page(self, project, page_key):
        pages = getattr(project, "pages", None)
        try:
            page_keys = tuple(pages.keys())
            page_index = page_keys.index(page_key)
        except (AttributeError, TypeError, ValueError):
            return None
        if page_index == 0:
            return None
        return self._snapshot_history_page(project, page_keys[page_index - 1])

    def _runtime_context_mode(self):
        value = self._context_param_value("context mode", CONTEXT_MODE_PAGE)
        return value if value in CONTEXT_MODES else CONTEXT_MODE_PAGE

    def _runtime_history_budget(self):
        value = self._context_param_value(
            "history token budget", _DEFAULT_HISTORY_TOKEN_BUDGET
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return _DEFAULT_HISTORY_TOKEN_BUDGET

    def _runtime_glossary_path(self):
        value = self._context_param_value("glossary path", "")
        try:
            path = os.fspath(value)
        except TypeError:
            return ""
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        return path.strip()

    def _runtime_glossary_mode(self):
        value = self._context_param_value(
            "glossary mode", GLOSSARY_MODE_MATCHING
        )
        return value if value in GLOSSARY_MODES else GLOSSARY_MODE_MATCHING

    def _context_param_value(self, key, default):
        params = getattr(self, "params", None)
        if not isinstance(params, dict):
            return default
        value = params.get(key, default)
        if isinstance(value, dict):
            return value.get("value", default)
        return value

    @staticmethod
    def _runtime_page_key(page_key):
        if isinstance(page_key, str):
            return page_key
        if page_key is None:
            return ""
        return str(page_key)
