"""Immutable, whole-page history selection for LLM translation requests."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Tuple

from .glossary import GlossaryEntry


HISTORY_LOW_WATER_RATIO = 0.60


class ContextAction(Enum):
    DISABLED = "disabled"
    EMPTY = "empty"
    REBUILD = "rebuild"
    REUSE = "reuse"
    GROW = "grow"
    EVICT = "evict"
    CONTEXT_RECOVERY = "context-recovery"


class ContextReason(Enum):
    HISTORY_DISABLED = "history-disabled"
    MISSING_PROJECT_PAGE = "missing-project-page"
    WINDOW_EMPTY = "window-empty"
    MISSING_LOAD_IDENTITY = "missing-load-identity"
    PROJECT_CHANGED = "project-changed"
    SETTINGS_CHANGED = "settings-changed"
    MISSING_PAGES = "missing-pages"
    NON_ADJACENT = "non-adjacent"
    SNAPSHOT_CHANGED = "snapshot-changed"
    PREVIOUS_INCOMPLETE = "previous-incomplete"
    OVERSIZED_PAGE = "oversized-page"


@dataclass(frozen=True)
class HistoryPage:
    page_key: str
    sources: Tuple[str, ...]
    translations: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedHistoryPage:
    snapshot: HistoryPage
    messages: Tuple[Tuple[str, str], ...]
    token_count: int

    @property
    def page_key(self) -> str:
        return self.snapshot.page_key


@dataclass(frozen=True)
class HistoryWindowKey:
    load_identity: object
    settings: Tuple[Tuple[str, object], ...]


@dataclass(frozen=True)
class HistoryWindow:
    key: HistoryWindowKey
    request_page_key: str
    history: Tuple[RenderedHistoryPage, ...]
    token_count: int


@dataclass(frozen=True)
class ContextDiagnostic:
    page_key: str
    action: ContextAction
    page_count: int
    token_count: int
    token_budget: int
    appended: int = 0
    evicted: int = 0
    rebuild_reason: Optional[ContextReason] = None

    def __str__(self) -> str:
        parts = [
            "LLM Context: page={}".format(self.page_key),
            "action={}".format(self.action.value),
            "pages={}".format(self.page_count),
            "tokens={}/{}".format(self.token_count, self.token_budget),
        ]
        if self.appended:
            parts.append("appended={}".format(self.appended))
        if self.evicted:
            parts.append("evicted={}".format(self.evicted))
        if self.rebuild_reason is not None:
            parts.append("reason={}".format(self.rebuild_reason.value))
        return ", ".join(parts)


@dataclass(frozen=True)
class RequestContext:
    history: Tuple[RenderedHistoryPage, ...]
    glossary: Tuple[GlossaryEntry, ...] = ()
    glossary_mode: str = ""
    history_budget: int = 0
    window_key: Optional[HistoryWindowKey] = None
    request_page_key: Optional[str] = None
    diagnostic: Optional[ContextDiagnostic] = None


def eligible_history_for_request(
    window: Optional[HistoryWindow],
    project: object,
    page_key: str,
    previous_page: Optional[HistoryPage],
    token_budget: int,
    rebuild_reason: Optional[ContextReason],
    snapshot_page: Callable[[str], Optional[HistoryPage]],
    render_page: Callable[[HistoryPage], RenderedHistoryPage],
) -> Tuple[Tuple[RenderedHistoryPage, ...], ContextDiagnostic]:
    """Select an immutable, chronological window of complete history pages."""

    token_budget = _nonnegative_budget(token_budget)
    if token_budget == 0:
        return _result(
            (), page_key, ContextAction.DISABLED, token_budget,
            rebuild_reason or ContextReason.HISTORY_DISABLED,
        )

    if rebuild_reason is not None:
        return _rebuild_history(
            project, page_key, token_budget, rebuild_reason, snapshot_page, render_page
        )

    if window is None:
        return _rebuild_history(
            project,
            page_key,
            token_budget,
            ContextReason.WINDOW_EMPTY,
            snapshot_page,
            render_page,
        )

    if _window_snapshot_changed(window, snapshot_page):
        return _rebuild_history(
            project,
            page_key,
            token_budget,
            ContextReason.SNAPSHOT_CHANGED,
            snapshot_page,
            render_page,
        )

    history = list(window.history)
    if previous_page is None or not _complete_snapshot(previous_page):
        return _result(
            tuple(history),
            page_key,
            ContextAction.REUSE,
            token_budget,
            ContextReason.PREVIOUS_INCOMPLETE,
        )

    rendered = _render_eligible(previous_page, render_page)
    if rendered is None:
        return _result(
            tuple(history),
            page_key,
            ContextAction.REUSE,
            token_budget,
            ContextReason.PREVIOUS_INCOMPLETE,
        )
    if rendered.token_count > token_budget:
        return _result(
            tuple(history),
            page_key,
            ContextAction.REUSE,
            token_budget,
            ContextReason.OVERSIZED_PAGE,
        )

    history.append(rendered)
    evicted = _evict_to_low_water(history, token_budget)
    action = ContextAction.EVICT if evicted else ContextAction.GROW
    return _result(
        tuple(history),
        page_key,
        action,
        token_budget,
        appended=1,
        evicted=evicted,
    )


def window_rebuild_reason(
    window: Optional[HistoryWindow],
    project: object,
    page_key: str,
    key: HistoryWindowKey,
) -> Optional[ContextReason]:
    """Return why a window cannot be safely reused for this project request."""

    if window is None:
        return ContextReason.WINDOW_EMPTY

    pages = getattr(project, "pages", None)
    if pages is None:
        return ContextReason.MISSING_PAGES
    try:
        page_keys = tuple(pages.keys())
    except AttributeError:
        return ContextReason.MISSING_PAGES
    if not page_keys:
        return ContextReason.MISSING_PAGES
    if page_key not in page_keys:
        return ContextReason.MISSING_PROJECT_PAGE

    project_identity = getattr(project, "load_identity", None)
    if project_identity is None or key is None or key.load_identity is None:
        return ContextReason.MISSING_LOAD_IDENTITY
    if (
        window.key.load_identity is not project_identity
        or key.load_identity is not project_identity
    ):
        return ContextReason.PROJECT_CHANGED
    if window.key.settings != key.settings:
        return ContextReason.SETTINGS_CHANGED
    if window.request_page_key not in page_keys:
        return ContextReason.MISSING_PAGES

    previous_index = page_keys.index(window.request_page_key)
    if previous_index + 1 >= len(page_keys) or page_keys[previous_index + 1] != page_key:
        return ContextReason.NON_ADJACENT
    return None


def recover_context_length(request_context: RequestContext) -> Optional[RequestContext]:
    """Drop one oldest immutable page after a provider rejects request length."""

    if request_context is None or not request_context.history:
        return None

    history = request_context.history[1:]
    diagnostic = ContextDiagnostic(
        page_key=request_context.request_page_key or "",
        action=ContextAction.CONTEXT_RECOVERY,
        page_count=len(history),
        token_count=_history_token_count(history),
        token_budget=_nonnegative_budget(request_context.history_budget),
        evicted=1,
    )
    return replace(request_context, history=history, diagnostic=diagnostic)


def _rebuild_history(
    project: object,
    page_key: str,
    token_budget: int,
    reason: ContextReason,
    snapshot_page: Callable[[str], Optional[HistoryPage]],
    render_page: Callable[[HistoryPage], RenderedHistoryPage],
) -> Tuple[Tuple[RenderedHistoryPage, ...], ContextDiagnostic]:
    page_keys = _project_page_keys(project)
    if page_key not in page_keys:
        return _result((), page_key, ContextAction.EMPTY, token_budget, reason)

    rebuild_limit = int(token_budget * HISTORY_LOW_WATER_RATIO)
    chosen_newest_first = []
    token_count = 0
    current_index = page_keys.index(page_key)
    for candidate_key in reversed(page_keys[:current_index]):
        snapshot = snapshot_page(candidate_key)
        rendered = _render_eligible(snapshot, render_page)
        if rendered is None or rendered.token_count > token_budget:
            continue
        if not chosen_newest_first:
            chosen_newest_first.append(rendered)
            token_count += rendered.token_count
            if token_count >= rebuild_limit:
                break
            continue
        if token_count + rendered.token_count > rebuild_limit:
            break
        chosen_newest_first.append(rendered)
        token_count += rendered.token_count

    history = tuple(reversed(chosen_newest_first))
    return _result(
        history,
        page_key,
        ContextAction.REBUILD if history else ContextAction.EMPTY,
        token_budget,
        reason,
    )


def _project_page_keys(project: object) -> Tuple[str, ...]:
    pages = getattr(project, "pages", None)
    if pages is None:
        return ()
    try:
        return tuple(pages.keys())
    except AttributeError:
        return ()


def _render_eligible(
    snapshot: Optional[HistoryPage],
    render_page: Callable[[HistoryPage], RenderedHistoryPage],
) -> Optional[RenderedHistoryPage]:
    if not _complete_snapshot(snapshot):
        return None
    rendered = render_page(snapshot)
    if not isinstance(rendered, RenderedHistoryPage):
        return None
    if rendered.page_key != snapshot.page_key or not _valid_token_count(rendered.token_count):
        return None
    return rendered


def _complete_snapshot(snapshot: Optional[HistoryPage]) -> bool:
    if not isinstance(snapshot, HistoryPage):
        return False
    if not snapshot.sources or len(snapshot.sources) != len(snapshot.translations):
        return False
    return all(
        isinstance(source, str)
        and isinstance(translation, str)
        and source.strip()
        and translation.strip()
        for source, translation in zip(snapshot.sources, snapshot.translations)
    )


def _window_snapshot_changed(
    window: HistoryWindow,
    snapshot_page: Callable[[str], Optional[HistoryPage]],
) -> bool:
    for rendered in window.history:
        current = snapshot_page(rendered.page_key)
        if current is not None and current != rendered.snapshot:
            return True
    return False


def _evict_to_low_water(history, token_budget: int) -> int:
    low_water = int(token_budget * HISTORY_LOW_WATER_RATIO)
    evicted = 0
    while len(history) > 1 and _history_token_count(history) > low_water:
        history.pop(0)
        evicted += 1
    return evicted


def _result(
    history: Tuple[RenderedHistoryPage, ...],
    page_key: str,
    action: ContextAction,
    token_budget: int,
    rebuild_reason: Optional[ContextReason] = None,
    appended: int = 0,
    evicted: int = 0,
) -> Tuple[Tuple[RenderedHistoryPage, ...], ContextDiagnostic]:
    return history, ContextDiagnostic(
        page_key=page_key,
        action=action,
        page_count=len(history),
        token_count=_history_token_count(history),
        token_budget=token_budget,
        appended=appended,
        evicted=evicted,
        rebuild_reason=rebuild_reason,
    )


def _history_token_count(history) -> int:
    return sum(page.token_count for page in history)


def _valid_token_count(token_count: object) -> bool:
    return isinstance(token_count, int) and not isinstance(token_count, bool) and token_count >= 0


def _nonnegative_budget(token_budget: object) -> int:
    if isinstance(token_budget, int) and not isinstance(token_budget, bool):
        return max(token_budget, 0)
    return 0
