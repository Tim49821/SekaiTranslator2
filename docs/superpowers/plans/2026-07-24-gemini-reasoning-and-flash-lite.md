# Gemini Reasoning Effort and Flash-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemini reasoning-effort dropdown and Gemini 3.5 Flash-Lite to both Google LLM translation and OCR.

**Architecture:** Keep the Gemini model and reasoning option constants in the shared LLM translator module, and reuse them from OCR. Extend the fixed-provider parameter builders and lazy metadata evaluator so only Google presets expose the dropdown, then add `reasoning_effort` to Google Chat Completions requests only when the user chooses a non-default value.

**Tech Stack:** Python 3, `unittest`, OpenAI Python client, existing Qt parameter renderer and lazy AST registry.

## Global Constraints

- The dropdown options are exactly `default`, `minimal`, `low`, `medium`, and `high`.
- Selecting `default` must omit `reasoning_effort` from the API request.
- The new model ID is exactly `GGL: gemini-3.5-flash-lite`.
- The setting is visible only in `LLM Google` and `LLM OCR Google`.
- Preserve all unrelated user changes in the working tree.

---

### Task 1: Shared model catalog and Google-only dropdown metadata

**Files:**
- Modify: `tests/test_llm_env.py`
- Modify: `tests/test_lazy_runtime.py`
- Modify: `modules/translators/trans_llm_api_json.py`
- Modify: `modules/lazy_registry.py`
- Modify: `modules/ocr/ocr_llm_api.py`

**Interfaces:**
- Produces: `GEMINI_REASONING_EFFORT_OPTIONS: List[str]`.
- Produces: `reasoning effort` selector metadata on the fixed Google translator and OCR classes.
- Produces: the shared `GGL: gemini-3.5-flash-lite` catalog entry.

- [ ] **Step 1: Write failing catalog and parameter tests**

Add imports for `GoogleLLMTranslator`, `OpenAILLMTranslator`, and `OpenAILLMOCR` where needed, then add:

```python
def test_gemini_35_flash_lite_model_is_shared(self):
    model = "GGL: gemini-3.5-flash-lite"
    self.assertIn(model, LLM_PROVIDER_MODEL_OPTIONS["Google"])
    self.assertIn(model, LLM_OCR_PROVIDER_MODEL_OPTIONS["Google"])

def test_reasoning_effort_is_exposed_only_by_google_presets(self):
    expected = ["default", "minimal", "low", "medium", "high"]
    self.assertEqual(
        GoogleLLMTranslator.params["reasoning effort"]["options"], expected
    )
    self.assertEqual(
        GoogleLLMOCR.params["reasoning effort"]["options"], expected
    )
    self.assertEqual(
        GoogleLLMTranslator.params["reasoning effort"]["value"], "default"
    )
    self.assertEqual(
        GoogleLLMOCR.params["reasoning effort"]["value"], "default"
    )
    self.assertNotIn("reasoning effort", OpenAILLMTranslator.params)
    self.assertNotIn("reasoning effort", OpenAILLMOCR.params)
```

Extend the subprocess script in `tests/test_lazy_runtime.py` with:

```python
expected_efforts = ["default", "minimal", "low", "medium", "high"]
llm_google = modules.TRANSLATORS.get("LLM Google")
llm_openai = modules.TRANSLATORS.get("LLM OpenAI")
ocr_google = modules.OCR.get("LLM OCR Google")
ocr_openai = modules.OCR.get("LLM OCR OpenAI")
assert llm_google.params["reasoning effort"]["options"] == expected_efforts
assert ocr_google.params["reasoning effort"]["options"] == expected_efforts
assert "reasoning effort" not in llm_openai.params
assert "reasoning effort" not in ocr_openai.params
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime -v`

Expected: FAIL because the new model and `reasoning effort` metadata do not exist.

- [ ] **Step 3: Implement the shared catalog and selector metadata**

In `trans_llm_api_json.py`, add the model and constant:

```python
GEMINI_REASONING_EFFORT_OPTIONS = [
    "default",
    "minimal",
    "low",
    "medium",
    "high",
]
```

Extend `_build_fixed_provider_params` with `include_reasoning_effort: bool = False`. When true, append:

```python
params["reasoning effort"] = {
    "type": "selector",
    "options": list(GEMINI_REASONING_EFFORT_OPTIONS),
    "value": "default",
    "description": "Controls Gemini reasoning depth. Default uses the model's native setting.",
}
```

Pass `True` only from `GoogleLLMTranslator`. Import the constant in `ocr_llm_api.py`, give its builder the same optional argument, and pass `True` only from `GoogleLLMOCR`.

Add the new model to `DEFAULT_LLM_PROVIDER_MODEL_OPTIONS["Google"]` in `lazy_registry.py`. Update `SafeEval.visit_Call` so `_build_fixed_provider_params` accepts three or four evaluated arguments and adds the same selector when the fourth argument is true.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime -v`

Expected: PASS.

- [ ] **Step 5: Commit the catalog and metadata change**

```bash
git add tests/test_llm_env.py tests/test_lazy_runtime.py modules/translators/trans_llm_api_json.py modules/lazy_registry.py modules/ocr/ocr_llm_api.py
git commit -m "feat: add Gemini reasoning settings"
```

---

### Task 2: Send reasoning effort in translator and OCR requests

**Files:**
- Create: `tests/test_gemini_reasoning.py`
- Modify: `modules/translators/trans_llm_api_json.py`
- Modify: `modules/ocr/ocr_llm_api.py`

**Interfaces:**
- Consumes: `GEMINI_REASONING_EFFORT_OPTIONS` and the `reasoning effort` parameter from Task 1.
- Produces: `apply_google_reasoning_effort(api_args: Dict, provider: str, effort: str) -> None`.

- [ ] **Step 1: Write failing translator request tests**

Create `tests/test_gemini_reasoning.py` with the translator test class below. The OCR class is added in Step 5.

```python
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.ocr.ocr_llm_api import GoogleLLMOCR
from modules.translators.trans_llm_api_json import GoogleLLMTranslator


def translation_completion():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"translations":[{"id":1,"translation":"안녕"}]}'
                )
            )
        ],
        usage=None,
    )


def ocr_completion():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="こんにちは"))],
        usage=None,
    )


class GeminiTranslatorReasoningTest(unittest.TestCase):
    def request_mock(self, effort):
        translator = GoogleLLMTranslator(
            "日本語",
            "한국어",
            raise_unsupported_lang=False,
            **{
                "apikey": "test-key",
                "reasoning effort": effort,
                "delay": 0,
            },
        )
        create = MagicMock(return_value=translation_completion())
        translator.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with (
            patch.object(translator, "_select_api_key", return_value="test-key"),
            patch.object(translator, "_initialize_client", return_value=True),
            patch.object(translator, "_respect_delay"),
        ):
            translator._request_translation("translate this")
        return create

    def test_sends_selected_reasoning_effort(self):
        create = self.request_mock("high")
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "high")

    def test_omits_reasoning_effort_for_default(self):
        create = self.request_mock("default")
        self.assertNotIn("reasoning_effort", create.call_args.kwargs)
```

- [ ] **Step 2: Run translator tests to verify RED**

Run: `python -m unittest tests.test_gemini_reasoning.GeminiTranslatorReasoningTest -v`

Expected: FAIL because translator requests do not include `reasoning_effort`.

- [ ] **Step 3: Implement translator request propagation**

Add a shared helper in `trans_llm_api_json.py`:

```python
def apply_google_reasoning_effort(
    api_args: Dict,
    provider: str,
    effort: str,
) -> None:
    if (
        provider == "Google"
        and effort in GEMINI_REASONING_EFFORT_OPTIONS
        and effort != "default"
    ):
        api_args["reasoning_effort"] = effort
```

Add a `reasoning_effort` property that safely returns `default` if the parameter is absent, then call the helper after constructing translator `api_args`.

- [ ] **Step 4: Run translator tests to verify GREEN**

Run: `python -m unittest tests.test_gemini_reasoning.GeminiTranslatorReasoningTest -v`

Expected: PASS.

- [ ] **Step 5: Write failing OCR request tests**

Append this class to `tests/test_gemini_reasoning.py`:

```python
class GeminiOCRReasoningTest(unittest.TestCase):
    def request_mock(self, effort):
        ocr = GoogleLLMOCR(
            **{
                "api_key": "test-key",
                "reasoning effort": effort,
                "delay": 0,
            }
        )
        create = MagicMock(return_value=ocr_completion())
        ocr.client = SimpleNamespace(
            api_key="test-key",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        with (
            patch.object(ocr, "_select_api_key", return_value="test-key"),
            patch.object(ocr, "_respect_delay"),
        ):
            self.assertEqual(ocr.ocr("encoded-image"), "こんにちは")
        return create

    def test_sends_selected_reasoning_effort(self):
        create = self.request_mock("high")
        self.assertEqual(create.call_args.kwargs["reasoning_effort"], "high")

    def test_omits_reasoning_effort_for_default(self):
        create = self.request_mock("default")
        self.assertNotIn("reasoning_effort", create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run OCR tests to verify RED**

Run: `python -m unittest tests.test_gemini_reasoning.GeminiOCRReasoningTest -v`

Expected: FAIL because OCR requests do not include `reasoning_effort`.

- [ ] **Step 7: Implement OCR request propagation**

Import `apply_google_reasoning_effort`, add the same safe `reasoning_effort` property, construct OCR request arguments in a dictionary, apply the helper, and call:

```python
response = self.client.chat.completions.create(**api_args)
```

- [ ] **Step 8: Run all reasoning tests to verify GREEN**

Run: `python -m unittest tests.test_gemini_reasoning -v`

Expected: PASS.

- [ ] **Step 9: Commit request propagation**

```bash
git add tests/test_gemini_reasoning.py modules/translators/trans_llm_api_json.py modules/ocr/ocr_llm_api.py
git commit -m "feat: send Gemini reasoning effort"
```

---

### Task 3: Documentation and full verification

**Files:**
- Modify: `doc/modules/translators.md`

**Interfaces:**
- Consumes: the model ID and dropdown behavior implemented in Tasks 1 and 2.
- Produces: user-facing documentation matching the shipped settings.

- [ ] **Step 1: Update translator documentation**

Add `GGL: gemini-3.5-flash-lite` to the Google model list and document:

```markdown
*   **reasoning effort:** Gemini reasoning depth (`default`, `minimal`, `low`, `medium`, or `high`). `default` preserves the selected model's native behavior.
```

State that the same control is available in `LLM OCR Google`.

- [ ] **Step 2: Run focused tests**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime tests.test_gemini_reasoning -v`

Expected: PASS with zero failures and errors.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS with zero failures and errors. If unrelated environment-dependent tests cannot run, record their exact errors and run every relevant focused test successfully.

- [ ] **Step 4: Inspect the final diff**

Run: `git diff --check && git diff --stat && git status --short`

Expected: no whitespace errors; only Gemini feature files and documentation are changed.

- [ ] **Step 5: Commit documentation**

```bash
git add doc/modules/translators.md
git commit -m "docs: document Gemini reasoning effort"
```
