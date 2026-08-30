"""Shared parameter schema for LLM translation context adapters."""

from typing import Dict


CONTEXT_MODE_PAGE = "page"
CONTEXT_MODE_HISTORY = "history"
CONTEXT_MODES = (CONTEXT_MODE_PAGE, CONTEXT_MODE_HISTORY)
CONTEXT_INVALIDATION_KEYS = {
    "context mode",
    "history token budget",
    "glossary path",
    "glossary mode",
    "model",
    "override model",
    "system_prompt",
    "system prompt presets",
    "style guide",
    "style guide presets",
    "thinking mode",
}


def build_llm_context_params() -> Dict:
    """Return a fresh parameter tree for one translator definition."""

    return {
        "context mode": {
            "type": "selector",
            "options": list(CONTEXT_MODES),
            "value": CONTEXT_MODE_PAGE,
            "display_name": "LLM context",
            "description": (
                "Use the current page alone or include eligible translated "
                "pages as history."
            ),
        },
        "history token budget": {
            "value": 4096,
            "display_name": "History token budget",
            "description": (
                "Estimated token budget reserved for complete prior-page "
                "examples."
            ),
        },
        "glossary path": {
            "type": "selector",
            "options": [""],
            "value": "",
            "editable": True,
            "path_selector": True,
            "path_filter": "*.json *.txt *.tsv",
            "size": "median",
            "display_name": "Glossary file",
            "description": (
                "Optional UTF-8 JSON, TXT, or TSV translation glossary."
            ),
        },
        "glossary mode": {
            "type": "selector",
            "options": ["matching", "all"],
            "value": "matching",
            "display_name": "Glossary mode",
            "description": (
                "Send matching current-page terms or the complete glossary."
            ),
        },
    }
