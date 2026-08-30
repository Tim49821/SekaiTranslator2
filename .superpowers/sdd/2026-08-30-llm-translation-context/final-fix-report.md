# Final fix report — PR 2

## Status and execution boundary

Implemented every Important production finding and every required test correction from `final-review-findings.md`. The two deferred-ruling items were not changed. Per the binding execution boundary, tests, compilation, lint, imports, `git diff --check`, and all other executable verification were **NOT RUN**. Self-review was limited to reading the final diff.

## Numbered findings

1. **Adjacent growth evicted below the hard budget.** `eligible_history_for_request()` now preserves the complete appended window whenever its total is at or below `token_budget`. Whole-oldest eviction toward the 60% low-water mark runs only after a hard-budget overflow. Added pure-helper and adapter-level below-hard-budget growth regressions while retaining the overflow regression.

2. **History committed before final page success.** Added a no-op `BaseTranslator._finalize_translation()` hook. `translate_textblk_lst()` now finalizes only after postprocess hooks, block assignment, exact-length validation, and empty/error-marker success evaluation; exceptions finalize false and propagate unchanged. `LLMContextAdapterMixin` owns a pending request context, clears stale/pending state at request start and on false/exception finalization, and commits only on true finalization. Remote and Gemma adapters stage only after structurally valid provider/worker results and only when commit intent is present. Context-length recovery stages the recovered immutable context. Added remote and Gemma coverage for success timing, empty/error-marker results, postprocess truncation, postprocess exceptions, pending cleanup, and recovered-context commit timing.

3. **Gemma wrong-length logging exposed response values.** Replaced the full malformed-response log with aggregate `response_type` and `item_count` fields. Added a wrong-length secret-sentinel regression asserting the sentinel never reaches logger calls.

4. **Gemini reasoning test used the obsolete request shape.** Updated `tests/test_gemini_reasoning.py` to pass a ready message list to `_request_translation()`.

5. **Glossary-free assertion was case-ineffective.** Updated the history-message assertion to compare against case-folded content.

6. **Adapter wiring lacked invalidation/reuse/growth coverage.** Added adapter-level tests for project load-identity invalidation, changed retained snapshots, adjacent oversized-page reuse, adjacent growth, and the below-hard-budget grow case.

7. **Pipeline branch/forwarding coverage was incomplete.** Expanded `tests/test_llm_context_pipeline.py` with focused fakes for strict-stage and serial direct translation, deferred low-VRAM translation, the public `MainWindow` → `ModuleManager` → `ImgtransThread` selected-block chain, and standalone success/error wrapper signal/dialog behavior. No model, network, or runtime work is invoked by the test design.

## Tests added or updated — NOT RUN

- `tests/test_llm_context_helpers.py`: hard-budget versus low-water adjacent growth.
- `tests/test_llm_project_context.py`: finalization ordering, false results, and exception propagation.
- `tests/test_llm_translation_context.py`: adapter invalidation/reuse/growth, remote/Gemma deferred commits and failure cleanup, recovered context, and secret-safe logging.
- `tests/test_llm_context_pipeline.py`: strict/direct, low-VRAM, public selected-block forwarding, and standalone wrapper behavior.
- `tests/test_gemini_reasoning.py`: ready message-list request fixture.

All tests above are **NOT RUN**. No existing or new test was executed.

## Files and commits

Production files:

- `modules/context/history.py`
- `modules/context/adapter.py`
- `modules/translators/base.py`
- `modules/translators/trans_llm_api_json.py`
- `modules/translators/trans_gemma4.py`

Test files are the five files listed in the preceding section.

- `98fa37dd` — `fix: finalize LLM history after page success` (production and regression tests)
- This report is recorded in a follow-up documentation commit.

## Self-review

Read the final diff and checked each numbered finding against the binding spec and final-review rulings. The commit path is framework-owned: adapters only stage provider/worker structural success, and the base text-block boundary decides whether staged state commits after user-visible postprocessing and assignment. Ordinary translators retain the same translation flow through the no-op base hook. The diff does not alter either deferred-ruling item.

## Concerns

Executable confidence is intentionally unavailable because the user reserved all verification. In particular, the expanded Qt/MainWindow pipeline tests and the new finalization regressions have not been imported or executed; any syntax, fixture, platform, or behavioral failure must be discovered by the user's verification run.
