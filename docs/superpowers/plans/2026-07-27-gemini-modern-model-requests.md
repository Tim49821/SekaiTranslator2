# Gemini 3.6 Flash and Modern Request Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gemini 3.6 Flash to Google translation and OCR while removing deprecated sampling parameters from current modern Gemini translation requests.

**Architecture:** Extend the existing shared Google model catalog and its lazy-registry fallback. Define an explicit set of modern Gemini API model IDs in the translator module, then conditionally add `temperature` and `top_p` only for models outside that set while leaving reasoning effort and JSON response handling unchanged.

**Tech Stack:** Python 3.8+, `unittest`, OpenAI-compatible Gemini Chat Completions, existing lazy AST registry.

## Global Constraints

- Add the exact UI model value `GGL: gemini-3.6-flash`.
- Expose the model in both `LLM Google` and `LLM OCR Google`.
- Keep `GGL: gemini-3.1-pro-preview` as the default.
- Omit `temperature` and `top_p` for exactly `gemini-3.6-flash` and `gemini-3.5-flash-lite`.
- Preserve existing request behavior for Gemini 3.1, Gemini 3 Flash Preview, unknown overrides, and non-Google providers.
- Preserve the existing `reasoning_effort`, JSON response format, and token-limit behavior.
- Preserve the user's unrelated `config/textstyles/default.json` change.

---

### Task 1: Shared Gemini 3.6 model catalog

**Files:**
- Modify: `tests/test_llm_env.py`
- Modify: `tests/test_lazy_runtime.py`
- Modify: `modules/translators/trans_llm_api_json.py`
- Modify: `modules/lazy_registry.py`

**Interfaces:**
- Produces: `GGL: gemini-3.6-flash` in `LLM_PROVIDER_MODEL_OPTIONS["Google"]`.
- Produces: matching eager and lazy model options for Google translation and OCR.

- [ ] **Step 1: Write failing catalog tests**

Add to `tests/test_llm_env.py`:

```python
def test_gemini_36_flash_model_is_shared(self):
    model = "GGL: gemini-3.6-flash"

    self.assertIn(model, LLM_PROVIDER_MODEL_OPTIONS["Google"])
    self.assertIn(model, LLM_OCR_PROVIDER_MODEL_OPTIONS["Google"])
    self.assertEqual(
        LLM_PROVIDER_DEFAULT_MODELS["Google"],
        "GGL: gemini-3.1-pro-preview",
    )
```

Import `LLM_PROVIDER_DEFAULT_MODELS` in that test module. Inside the subprocess script in `tests/test_lazy_runtime.py`, append:

```python
gemini_36 = "GGL: gemini-3.6-flash"
assert gemini_36 in llm_google.params["model"]["options"]
assert gemini_36 in ocr_google.params["model"]["options"]
```

- [ ] **Step 2: Run catalog tests to verify RED**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime -v`

Expected: FAIL because `GGL: gemini-3.6-flash` is absent from eager and lazy catalogs.

- [ ] **Step 3: Implement the shared catalog entry**

Add the new model as the first Google option in `trans_llm_api_json.py`:

```python
"Google": [
    "GGL: gemini-3.6-flash",
    "GGL: gemini-3.1-pro-preview",
    "GGL: gemini-3.5-flash-lite",
    "GGL: gemini-3-flash-preview",
    "GGL: gemini-3.1-flash-lite",
],
```

Make the identical addition to `DEFAULT_LLM_PROVIDER_MODEL_OPTIONS["Google"]` in `modules/lazy_registry.py`. Do not change either default-model mapping.

- [ ] **Step 4: Run catalog tests to verify GREEN**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime -v`

Expected: PASS.

- [ ] **Step 5: Commit the catalog change**

```bash
git add tests/test_llm_env.py tests/test_lazy_runtime.py modules/translators/trans_llm_api_json.py modules/lazy_registry.py
git commit -m "feat: add Gemini 3.6 Flash model"
```

---

### Task 2: Modern Gemini translation request fields

**Files:**
- Modify: `tests/test_gemini_reasoning.py`
- Modify: `modules/translators/trans_llm_api_json.py`

**Interfaces:**
- Produces: `GEMINI_MODELS_WITHOUT_SAMPLING_PARAMETERS: set[str]` containing `gemini-3.6-flash` and `gemini-3.5-flash-lite`.
- Consumes: the provider-stripped `model_name` already resolved by `_request_translation`.

- [ ] **Step 1: Generalize the request test helper**

Change `GeminiTranslatorReasoningTest.request_mock` to accept a model and include it in constructor parameters:

```python
def request_mock(
    self,
    effort,
    model="GGL: gemini-3.1-pro-preview",
    override_model="",
):
    translator = IsolatedGoogleLLMTranslator(
        "日本語",
        "한국어",
        raise_unsupported_lang=False,
        **{
            "apikey": "test-key",
            "model": model,
            "override model": override_model,
            "reasoning effort": effort,
            "delay": 0,
        },
    )
```

Keep the existing mock client and patch contexts unchanged.

- [ ] **Step 2: Write failing request-format tests**

Add these tests to `GeminiTranslatorReasoningTest`:

```python
def test_gemini_36_omits_deprecated_sampling_parameters(self):
    create = self.request_mock("high", model="GGL: gemini-3.6-flash")

    self.assertNotIn("temperature", create.call_args.kwargs)
    self.assertNotIn("top_p", create.call_args.kwargs)
    self.assertEqual(create.call_args.kwargs["reasoning_effort"], "high")

def test_gemini_35_flash_lite_omits_deprecated_sampling_parameters(self):
    create = self.request_mock(
        "default",
        model="GGL: gemini-3.5-flash-lite",
    )

    self.assertNotIn("temperature", create.call_args.kwargs)
    self.assertNotIn("top_p", create.call_args.kwargs)

def test_legacy_gemini_keeps_sampling_parameters(self):
    create = self.request_mock(
        "default",
        model="GGL: gemini-3.1-pro-preview",
    )

    self.assertEqual(create.call_args.kwargs["temperature"], 0.1)
    self.assertEqual(create.call_args.kwargs["top_p"], 1.0)

def test_modern_override_model_omits_sampling_parameters(self):
    create = self.request_mock(
        "default",
        override_model="gemini-3.6-flash",
    )

    self.assertNotIn("temperature", create.call_args.kwargs)
    self.assertNotIn("top_p", create.call_args.kwargs)
```

- [ ] **Step 3: Run request tests to verify RED**

Run: `python -m unittest tests.test_gemini_reasoning.GeminiTranslatorReasoningTest -v`

Expected: the modern-model omission tests FAIL because all translation requests currently include `temperature` and `top_p`; the legacy retention test passes.

- [ ] **Step 4: Implement explicit modern-model request handling**

Add near the existing Gemini reasoning options:

```python
GEMINI_MODELS_WITHOUT_SAMPLING_PARAMETERS = {
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
}
```

Build the common translation arguments first:

```python
api_args = {
    "model": model_name,
    "messages": messages,
    "max_tokens": self.max_tokens,
}
if not (
    self.provider == "Google"
    and model_name in GEMINI_MODELS_WITHOUT_SAMPLING_PARAMETERS
):
    api_args["temperature"] = self.temperature
    api_args["top_p"] = self.top_p
```

Leave `apply_google_reasoning_effort`, response-format construction, and provider-specific penalties in their current order.

- [ ] **Step 5: Run request tests to verify GREEN**

Run: `python -m unittest tests.test_gemini_reasoning.GeminiTranslatorReasoningTest -v`

Expected: PASS.

- [ ] **Step 6: Run focused Gemini tests**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime tests.test_gemini_reasoning -v`

Expected: PASS.

- [ ] **Step 7: Commit request-format handling**

```bash
git add tests/test_gemini_reasoning.py modules/translators/trans_llm_api_json.py
git commit -m "fix: modernize latest Gemini requests"
```

---

### Task 3: Documentation and full verification

**Files:**
- Modify: `doc/modules/translators.md`

**Interfaces:**
- Consumes: the shared Gemini 3.6 model option and modern request-format behavior.
- Produces: user-facing model and compatibility documentation.

- [ ] **Step 1: Update the Google model documentation**

Replace the Google model line with:

```markdown
*   Models shown in the UI: `GGL: gemini-3.6-flash`, `GGL: gemini-3.1-pro-preview`, `GGL: gemini-3.5-flash-lite`, `GGL: gemini-3-flash-preview`, `GGL: gemini-3.1-flash-lite`
```

Add below the reasoning-effort field documentation:

```markdown
*   **Modern Gemini requests:** `gemini-3.6-flash` and `gemini-3.5-flash-lite` omit deprecated `temperature` and `top_p` fields automatically.
```

- [ ] **Step 2: Run the focused tests**

Run: `python -m unittest tests.test_llm_env tests.test_lazy_runtime tests.test_gemini_reasoning -v`

Expected: PASS with zero failures and errors.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS with zero failures and errors.

- [ ] **Step 4: Inspect final repository state**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors; only `doc/modules/translators.md` remains as a feature change, alongside the pre-existing user change in `config/textstyles/default.json`.

- [ ] **Step 5: Commit documentation**

```bash
git add doc/modules/translators.md
git commit -m "docs: add Gemini 3.6 Flash"
```
