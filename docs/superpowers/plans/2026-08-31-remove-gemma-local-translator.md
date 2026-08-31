# Remove Gemma Local Translator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Gemma 4 local translator from the runtime catalog and remove its dedicated implementation, setup helper, tests, and current user documentation.

**Architecture:** Translator discovery is file-driven, so deleting the registered `trans_gemma4` module removes the catalog entry without a replacement shim. Config loading migrates a legacy Gemma selection to the default Google translator and removes stale Gemma parameters. Gemma-only worker/setup code and tests are deleted, while shared local-model preparation and remote LLM context behavior remain intact. Historical specs, plans, and completed engineering reports stay as archival records.

**Tech Stack:** Python 3, unittest/pytest, lazy module registry, Markdown documentation

**Spec:** `doc/modules/translators.md` (current behavior being retired)

## Global Constraints

- Preserve all remote OpenAI-compatible LLM translators and shared translation-context/glossary behavior.
- Preserve generic Hugging Face local-model preparation utilities and tests.
- Do not delete historical documents under `docs/superpowers/` or `.superpowers/`.
- Do not delete downloaded model/runtime data from `data/models/`; only remove repository-owned source files.

---

### Task 1: Retire the Gemma runtime surface

**Files:**
- Modify: `tests/test_local_translators.py`
- Modify: `tests/test_lazy_runtime.py`
- Delete: `modules/translators/trans_gemma4.py`
- Delete: `modules/translators/gemma4_worker.py`
- Delete: `scripts/setup_gemma4_runtime.py`
- Modify: `tests/test_llm_translation_context.py`
- Modify: `utils/config.py`
- Modify: `tests/test_config_template.py`
- Modify: `doc/modules/translators.md`

**Interfaces:**
- Consumes: file-driven translator discovery from `modules.base.init_translator_registries()`
- Produces: a translator registry without the key `Gemma 4 E4B-it`; remote LLM translator APIs remain unchanged

- [x] **Step 1: Write the failing catalog test**

Change the registration expectation to exercise the real registry:

```python
class LocalTranslatorRegistrationTest(unittest.TestCase):
    def test_retired_local_translators_are_not_registered(self):
        init_translator_registries()

        self.assertNotIn("Gemma 4 E4B-it", TRANSLATORS.module_dict)
        self.assertNotIn("NLLB-200 distilled 1.3B", TRANSLATORS.module_dict)
        self.assertNotIn("Qwen3.5 9B GGUF", TRANSLATORS.module_dict)
```

- [x] **Step 2: Run the catalog test and verify RED**

Run: `python -m pytest tests/test_local_translators.py::LocalTranslatorRegistrationTest::test_retired_local_translators_are_not_registered -q`

Expected: FAIL because `Gemma 4 E4B-it` is still discovered and registered.

- [x] **Step 3: Remove Gemma-owned implementation and tests**

Delete the three Gemma-owned source files. In `tests/test_local_translators.py`, remove Gemma/setup imports, Gemma-only fake classes, `GGUFSetupRuntimeTest`, and `GemmaTranslatorTest`, while retaining shared registry, remote LLM catalog, base-module, and Hugging Face preparation tests. In `tests/test_llm_translation_context.py`, remove the `Gemma4E4BTranslator` import and `GemmaContextAdapterTest` while retaining shared and remote context tests. In `tests/test_lazy_runtime.py`, remove the Gemma metadata assertion and retain the remote LLM metadata assertion.

- [x] **Step 4: Update current translator documentation**

In `doc/modules/translators.md`, remove Gemma from the current translator summary and context description, replace the Gemma-worker failure wording with provider-neutral wording, and remove the entire `Local GGUF Translators` section and its table-of-contents entry.

- [x] **Step 5: Verify GREEN and regressions**

Run:

```bash
python -m pytest tests/test_local_translators.py tests/test_lazy_runtime.py tests/test_llm_translation_context.py tests/test_config_template.py -q
```

Expected: PASS with no imports, registry entries, or current documentation for the removed translator.

- [x] **Step 6: Verify active-reference cleanup**

Run:

```bash
rg -n -i 'gemma|gemma4|gemma-4|trans_gemma4|setup_gemma4_runtime' modules scripts tests doc
```

Expected: only compatibility migration code/tests, the registry regression assertion, and archival references such as the completed improvement tracker may remain; no translator implementation, import, or current translator-documentation references remain.

## Execution Notes

- The environment did not provide `pytest`, so the equivalent `python -m unittest` selectors were used.
- The removal-specific, config-migration, and directly affected suites passed 63 tests.
- Full discovery ran 259 tests with 254 passing. The remaining lazy-registry failure and four `OCRThread.ocr` setter errors reproduce unchanged on the original `main` checkout and are outside this removal.
