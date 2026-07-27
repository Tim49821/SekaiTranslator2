# LLM System Prompt Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-specific system-prompt preset management to every remote LLM translator and ship a strict Japanese doujin/manga-to-Korean translation preset.

**Architecture:** Generalize the existing Qt style-guide manager into a metadata-driven preset manager that synchronizes one visible preset collection with one hidden active parameter. Define built-in system-prompt presets in the remote LLM translator module; the existing per-translator parameter deep copies and `translator_params` persistence keep providers isolated without a new storage layer.

**Tech Stack:** Python 3.8+, `unittest`, Qt via `qtpy`, OpenAI-compatible Chat Completions, existing module parameter/configuration system.

## Global Constraints

- Store presets independently for `LLM OpenAI`, `LLM Google`, `LLM Grok`, `LLM OpenRouter`, and `LLM Studio`.
- Preserve every existing customized `system_prompt` as that provider's `Default` prompt when no selected custom preset is active.
- Keep `system_prompt` as the parameter consumed by API requests; do not change request batching, response parsing, or endpoints.
- Ship exactly two built-in prompt names: `Default` and `Japanese Doujin/Manga → Korean`.
- Keep `Default` undeletable and prevent deletion of the final remaining preset.
- Do not change LLM OCR prompts or local Gemma saved preset data.
- Preserve the user's unrelated `config/textstyles/default.json` modification.
- Support Python 3.8; do not introduce PEP 604 union syntax or built-in generic annotations that require a newer interpreter.

---

### Task 1: Provider-specific built-in prompt definitions

**Files:**
- Create: `tests/test_llm_prompt_presets.py`
- Modify: `modules/translators/trans_llm_api_json.py:103-205`

**Interfaces:**
- Produces: `DEFAULT_LLM_SYSTEM_PROMPT: str`.
- Produces: `JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT: str`.
- Produces: `LLM_SYSTEM_PROMPT_PRESETS: Dict[str, str]`.
- Produces: the hidden active parameter `system_prompt` and visible manager parameter `system prompt presets` in every fixed-provider parameter copy.

- [ ] **Step 1: Write failing provider and prompt-content tests**

Create `tests/test_llm_prompt_presets.py` with an offscreen Qt environment and add:

```python
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.translators.trans_llm_api_json import (
    GoogleLLMTranslator,
    GrokLLMTranslator,
    JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT,
    LLMStudioTranslator,
    OpenAILLMTranslator,
    OpenRouterLLMTranslator,
)


class LLMPromptPresetDefinitionTest(unittest.TestCase):
    provider_classes = (
        OpenAILLMTranslator,
        GoogleLLMTranslator,
        GrokLLMTranslator,
        OpenRouterLLMTranslator,
        LLMStudioTranslator,
    )

    def test_every_provider_has_independent_builtin_prompt_presets(self):
        preset_values = [
            provider.params["system prompt presets"]["value"]
            for provider in self.provider_classes
        ]

        for provider in self.provider_classes:
            self.assertTrue(provider.params["system_prompt"]["hidden"])
            self.assertEqual(
                list(provider.params["system prompt presets"]["value"]["styles"]),
                ["Default", "Japanese Doujin/Manga → Korean"],
            )
        for index, value in enumerate(preset_values):
            for other in preset_values[index + 1:]:
                self.assertIsNot(value, other)
                self.assertIsNot(value["styles"], other["styles"])

    def test_manga_prompt_requires_contextual_faithful_korean_json(self):
        prompt = JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT

        self.assertIn("Japanese", prompt)
        self.assertIn("natural Korean", prompt)
        self.assertIn("shared page context", prompt)
        self.assertIn("speech levels", prompt)
        self.assertIn("SFX", prompt)
        self.assertIn("vertical", prompt)
        self.assertIn("Do not censor", prompt)
        self.assertIn("exactly one", prompt)
        self.assertIn("numeric id", prompt)
        self.assertIn("valid JSON", prompt)
```

- [ ] **Step 2: Run the definition tests to verify RED**

Run: `python -m unittest tests.test_llm_prompt_presets.LLMPromptPresetDefinitionTest -v`

Expected: FAIL because the manga prompt constant and `system prompt presets` parameter do not exist.

- [ ] **Step 3: Add prompt constants and preset metadata**

In `modules/translators/trans_llm_api_json.py`, extract the existing system prompt into `DEFAULT_LLM_SYSTEM_PROMPT`. Add `JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT` with these exact behavioral requirements:

```python
DEFAULT_LLM_SYSTEM_PROMPT = (
    "You are an expert translator. Your task is to accurately translate the given text snippets. "
    "You MUST provide the output strictly in the specified JSON format, without any additional "
    "explanations or markdown formatting. The JSON object must have a single key 'translations', "
    "which is a list of objects, each with an 'id' (integer) and a 'translation' (string).\n\n"
    "Example Output Schema:\n"
    '{"translations": [{"id": 1, "translation": "Translated text here."}]}'
)

JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT = (
    "You are a professional translator specializing in Japanese doujinshi and manga translated "
    "into natural Korean. Treat every supplied text cell as ordered shared page context, using the "
    "other cells to resolve speaker continuity, character voice, relationships, terminology, and "
    "tone. Produce concise Korean suitable for speech balloons while faithfully preserving meaning, "
    "character-specific speech levels, honorifics, emotional intensity, ambiguity, punctuation, and "
    "intentional repetition. Render SFX, onomatopoeia, and mimetic language naturally and compactly. "
    "Correct only obvious OCR noise, including Japanese vertical text that is clearly misordered, and "
    "never invent missing dialogue. Do not censor, arbitrarily soften, embellish, summarize, or add "
    "translator notes. Return exactly one translation for every requested numeric id. Output only a "
    "valid JSON object whose single 'translations' array contains objects with 'id' and 'translation'; "
    "do not output markdown, explanations, alternatives, or repeated source text."
)

LLM_SYSTEM_PROMPT_PRESETS = {
    "Default": DEFAULT_LLM_SYSTEM_PROMPT,
    "Japanese Doujin/Manga → Korean": JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT,
}
```

Change `system_prompt` to use `DEFAULT_LLM_SYSTEM_PROMPT` and set `hidden: True`. Directly after it, add:

```python
"system prompt presets": {
    "type": "preset_manager",
    "value": {
        "selected": "Default",
        "styles": deepcopy(LLM_SYSTEM_PROMPT_PRESETS),
    },
    "default_presets": deepcopy(LLM_SYSTEM_PROMPT_PRESETS),
    "target_param": "system_prompt",
    "add_title": "Add system prompt",
    "name_label": "Prompt name:",
    "add_button": "Add prompt",
    "replace_button": "Replace prompt",
    "delete_button": "Delete prompt",
    "description": "Select, add, replace, or delete reusable system prompts for this LLM provider.",
},
```

- [ ] **Step 4: Run the definition tests to verify GREEN**

Run: `python -m unittest tests.test_llm_prompt_presets.LLMPromptPresetDefinitionTest -v`

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit provider definitions**

```bash
git add tests/test_llm_prompt_presets.py modules/translators/trans_llm_api_json.py
git commit -m "feat: define LLM system prompt presets"
```

---

### Task 2: Generic preset-manager widget and compatibility

**Files:**
- Modify: `tests/test_llm_prompt_presets.py`
- Modify: `ui/module_parse_widgets.py:133-247`

**Interfaces:**
- Consumes: a parameter dictionary whose manager metadata has `type: "preset_manager"`, `target_param: str`, and `value: {"selected": str, "styles": Dict[str, str]}`.
- Produces: `ParamPresetManager(QWidget)` with `add_preset(name: str) -> bool`, `replace_selected_preset() -> None`, and `delete_selected_preset() -> bool`.
- Preserves: `ParamStyleGuideManager` as a compatibility subclass using target parameter `style guide` and the existing style-specific labels.

- [ ] **Step 1: Write failing widget state and action tests**

Append to `tests/test_llm_prompt_presets.py`:

```python
from copy import deepcopy

from qtpy.QtWidgets import QApplication

from ui.module_parse_widgets import ParamPresetManager, ParamWidget


def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class PromptPresetManagerTest(unittest.TestCase):
    def setUp(self):
        self.app = ensure_app()
        self.params = deepcopy(OpenAILLMTranslator.params)

    def make_manager(self):
        return ParamPresetManager(
            "system prompt presets",
            self.params["system prompt presets"],
            self.params,
        )

    def test_existing_custom_prompt_becomes_default(self):
        self.params["system_prompt"]["value"] = "My existing custom prompt"

        manager = self.make_manager()

        self.assertEqual(manager.presets["Default"], "My existing custom prompt")
        self.assertEqual(manager.editor.text(), "My existing custom prompt")
        manager.close()

    def test_reloading_nondefault_selection_preserves_default(self):
        state = self.params["system prompt presets"]["value"]
        original_default = state["styles"]["Default"]
        state["selected"] = "Japanese Doujin/Manga → Korean"
        self.params["system_prompt"]["value"] = state["styles"][state["selected"]]

        manager = self.make_manager()

        self.assertEqual(manager.presets["Default"], original_default)
        self.assertEqual(manager.selected, "Japanese Doujin/Manga → Korean")
        manager.close()

    def test_selecting_preset_updates_active_system_prompt(self):
        manager = self.make_manager()

        manager.on_selected_preset_changed("Japanese Doujin/Manga → Korean")

        self.assertEqual(
            self.params["system_prompt"]["value"],
            JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT,
        )
        self.assertEqual(manager.editor.text(), JAPANESE_MANGA_KOREAN_SYSTEM_PROMPT)
        manager.close()

    def test_invalid_saved_state_falls_back_to_builtins(self):
        self.params["system prompt presets"]["value"] = {
            "selected": "Missing",
            "styles": [],
        }

        manager = self.make_manager()

        self.assertEqual(
            list(manager.presets),
            ["Default", "Japanese Doujin/Manga → Korean"],
        )
        self.assertEqual(manager.selected, "Default")
        manager.close()

    def test_add_replace_delete_updates_state_and_protects_default(self):
        manager = self.make_manager()
        manager.editor.setText("First custom prompt")

        self.assertTrue(manager.add_preset("My Prompt"))
        manager.editor.setText("Replacement prompt")
        manager.replace_selected_preset()
        self.assertEqual(manager.presets["My Prompt"], "Replacement prompt")
        self.assertTrue(manager.delete_selected_preset())
        self.assertNotIn("My Prompt", manager.presets)
        self.assertEqual(manager.selected, "Default")
        self.assertFalse(manager.delete_selected_preset())
        manager.close()

    def test_param_widget_builds_system_prompt_manager(self):
        widget = ParamWidget(self.params)

        managers = widget.findChildren(ParamPresetManager)

        self.assertEqual(len(managers), 1)
        self.assertEqual(managers[0].target_param, "system_prompt")
        widget.close()
```

- [ ] **Step 2: Run widget tests to verify RED**

Run: `python -m unittest tests.test_llm_prompt_presets.PromptPresetManagerTest -v`

Expected: FAIL because `ParamPresetManager`, the generic parser branch, and its public state methods do not exist.

- [ ] **Step 3: Implement the generic manager**

Refactor `ParamStyleGuideManager` logic into `ParamPresetManager`:

```python
class ParamPresetManager(QWidget):
    paramwidget_edited = Signal(str, object)

    def __init__(self, param_key, param_dict, all_params, scrollWidget=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.param_dict = param_dict
        self.all_params = all_params
        self.target_param = param_dict.get("target_param", "style guide")
        self.protected_name = param_dict.get("protected_name", "Default")
        self._syncing = False

        value = param_dict.get("value")
        value = value if isinstance(value, dict) else {}
        default_presets = param_dict.get("default_presets")
        self.presets = dict(default_presets) if isinstance(default_presets, dict) else {}
        saved_presets = value.get("styles")
        if isinstance(saved_presets, dict):
            self.presets.update(saved_presets)
        if self.protected_name not in self.presets:
            self.presets[self.protected_name] = ""
        self.selected = value.get("selected") or self.protected_name
        if self.selected not in self.presets:
            self.selected = self.protected_name

        active_text = self._current_target_text()
        if active_text:
            if self.selected == self.protected_name:
                self.presets[self.protected_name] = active_text
            elif active_text != self.presets.get(self.selected):
                self.presets[self.selected] = active_text
        self.styles = self.presets
```

Build the selector, editor, and buttons using metadata labels with the current style-guide strings as defaults. Replace hard-coded `style guide` emissions with `self.target_param`. Make `_state()` return the unchanged serialized shape `{"selected": ..., "styles": ...}` and update both `all_params[self.param_key]["value"]` and `all_params[self.target_param]["value"]` before emitting their respective signals.

Add the public state operations:

```python
def add_preset(self, name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    self.selected = name
    self.presets[name] = self.editor.text()
    self._set_selector_items()
    self._emit_state()
    self._emit_active_preset()
    return True

def replace_selected_preset(self) -> None:
    self.presets[self.selected] = self.editor.text()
    self._emit_state()
    self._emit_active_preset()

def delete_selected_preset(self) -> bool:
    if self.selected == self.protected_name or len(self.presets) <= 1:
        return False
    self.presets.pop(self.selected, None)
    self.selected = self.protected_name if self.protected_name in self.presets else next(iter(self.presets))
    self._set_selector_items()
    self._set_editor_text(self.presets[self.selected])
    self._emit_state()
    self._emit_active_preset()
    return True
```

Keep button slots thin: the add slot opens `QInputDialog` and calls `add_preset(name)` when accepted; replace and delete slots call the corresponding public method. Provide `ParamStyleGuideManager(ParamPresetManager)` as a compatibility subclass that supplies the prior style-guide labels and `target_param` defaults before calling `super().__init__`.

Add a parser branch next to `style_guide_manager`:

```python
elif param_type == "preset_manager":
    param_widget = ParamPresetManager(
        param_key,
        param_dict,
        params,
        scrollWidget=scrollWidget,
    )
```

- [ ] **Step 4: Run prompt-manager and Gemma regression tests to verify GREEN**

Run: `python -m unittest tests.test_llm_prompt_presets tests.test_local_translators -v`

Expected: PASS, including the new prompt-manager tests and existing Gemma style-guide behavior.

- [ ] **Step 5: Commit the preset-manager integration**

```bash
git add tests/test_llm_prompt_presets.py ui/module_parse_widgets.py
git commit -m "feat: manage provider system prompt presets"
```

---

### Task 3: Documentation and full verification

**Files:**
- Modify: `doc/modules/translators.md:38-65`

**Interfaces:**
- Documents: provider-local prompt selection, editing, add/replace/delete behavior, compatibility with existing custom prompts, and the built-in manga preset.

- [ ] **Step 1: Update translator documentation**

Add these settings under OpenAI-compatible LLM providers:

```markdown
*   **system prompt presets:** Provider-specific dropdown and editor for reusable system prompts. Use Add, Replace, and Delete to manage presets; `Default` cannot be deleted. Existing custom system prompts are retained as that provider's default.
*   **Japanese Doujin/Manga → Korean:** Built-in system-prompt preset for page-context-aware Japanese manga translation with natural Korean dialogue, stable character voice and speech levels, compact SFX handling, conservative OCR repair, and strict JSON output.
```

- [ ] **Step 2: Run focused feature tests**

Run: `python -m unittest tests.test_llm_prompt_presets tests.test_local_translators -v`

Expected: PASS with no failures or errors.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS with no failures or errors. If an environment-specific optional dependency causes a skip, record the skip count and confirm it is unrelated to this feature.

- [ ] **Step 4: Run syntax and diff checks**

Run: `python -m py_compile modules/translators/trans_llm_api_json.py ui/module_parse_widgets.py tests/test_llm_prompt_presets.py`

Expected: exit code 0.

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 5: Commit documentation**

```bash
git add doc/modules/translators.md docs/superpowers/plans/2026-07-28-llm-system-prompt-presets.md
git commit -m "docs: explain LLM system prompt presets"
```
