# Free/Paid API Key Pools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add independently selectable Free and Paid round-robin API-key pools to every remote LLM translator and LLM OCR module while preserving existing keys as the default Free pool.

**Architecture:** \`utils.env\` owns tier normalization, list parsing, environment naming, persistence, sanitization, and legacy fallback. Shared translator and OCR implementations expose the new settings and consume the selected pool, so fixed-provider modules inherit the behavior without duplication.

**Tech Stack:** Python 3.11, Qt parameter-schema UI, \`unittest\`, existing OpenAI-compatible clients.

## Global Constraints

- Translation and OCR select tiers independently.
- Only the selected pool is used; there is no cross-tier fallback.
- Both pools accept semicolon- or newline-separated keys and rotate in order.
- Existing single and multiple keys become the default Free pool.
- Secrets remain in \`.env\` and are removed from saved configuration JSON.
- Local LLM Studio and Ollama behavior remains unchanged.
- Preserve unrelated working-tree changes, especially \`config/textstyles/default.json\`.

---

### Task 1: Tier-aware environment storage

**Files:**
- Modify: \`utils/env.py\`
- Modify: \`tests/test_llm_env.py\`

**Interfaces:**
- Produces: \`normalize_llm_api_key_tier(tier: str) -> str\`
- Produces: \`parse_llm_api_keys(value: str) -> List[str]\`
- Produces: \`get_llm_api_key_pool(provider: str, tier: str = "Free", for_ocr: bool = False) -> List[str]\`
- Extends persistence and sanitization for \`free_api_keys\` and \`paid_api_keys\`.

- [ ] **Step 1: Write failing environment tests**

Add imports for the new helpers and tests equivalent to:

\`\`\`python
def test_tier_pools_are_independent_and_deduplicated(self):
    env = {
        "BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS": "free-a;free-b\nfree-a",
        "BALLOONTRANS_LLM_OPENAI_PAID_API_KEYS": "paid-a;paid-b",
        "BALLOONTRANS_LLM_OCR_OPENAI_FREE_API_KEYS": "ocr-free",
    }
    with patch("utils.env.load_dotenv"), patch.dict(os.environ, env, clear=True):
        self.assertEqual(get_llm_api_key_pool("OpenAI", "Free"), ["free-a", "free-b"])
        self.assertEqual(get_llm_api_key_pool("OpenAI", "Paid"), ["paid-a", "paid-b"])
        self.assertEqual(
            get_llm_api_key_pool("OpenAI", "Free", for_ocr=True),
            ["ocr-free"],
        )

def test_legacy_keys_form_default_free_pool(self):
    env = {
        "BALLOONTRANS_LLM_GOOGLE_API_KEY": "legacy-single",
        "BALLOONTRANS_LLM_GOOGLE_API_KEYS": "legacy-a;legacy-single;legacy-b",
    }
    with patch("utils.env.load_dotenv"), patch.dict(os.environ, env, clear=True):
        self.assertEqual(
            get_llm_api_key_pool("Google", "unknown"),
            ["legacy-single", "legacy-a", "legacy-b"],
        )
        self.assertEqual(get_llm_api_key_pool("Google", "Paid"), [])
\`\`\`

Extend the persistence fixture with Free/Paid fields for translation and OCR. Assert tier-aware variables are written and all secret fields sanitize to empty strings. Add a legacy-only fixture asserting legacy single/multiple values are combined into the new Free variable.

- [ ] **Step 2: Run tests and verify RED**

Run: \`.venv/bin/python -m unittest tests.test_llm_env\`

Expected: import or assertion failures because tier helpers and fields do not exist.

- [ ] **Step 3: Implement the environment layer**

Add:

\`\`\`python
from typing import Dict, List, Mapping, Optional, Tuple

LLM_API_KEY_TIERS = ("Free", "Paid")

def normalize_llm_api_key_tier(tier: str) -> str:
    return "Paid" if isinstance(tier, str) and tier.strip().lower() == "paid" else "Free"

def parse_llm_api_keys(value: str) -> List[str]:
    keys, seen = [], set()
    if not isinstance(value, str):
        return keys
    for key in value.replace("\n", ";").split(";"):
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys
\`\`\`

Add \`_primary_tier_env(provider, tier, for_ocr=False)\` using the design's exact variable names. \`get_llm_api_key_pool\` reads the selected tier variable; Free falls back to legacy single followed by legacy multiple values. Update \`_collect_llm_api_keys\` to persist new fields and convert loaded legacy fields into a Free pool. Update \`sanitize_llm_api_keys\` to clear all old and new secret fields.

- [ ] **Step 4: Run tests and verify GREEN**

Run: \`.venv/bin/python -m unittest tests.test_llm_env\`

Expected: all tests pass.

- [ ] **Step 5: Commit**

\`\`\`bash
git add utils/env.py tests/test_llm_env.py
git commit -m "feat: add tiered LLM API key storage"
\`\`\`

---

### Task 2: Translator tier selection

**Files:**
- Create: \`tests/test_llm_api_key_pools.py\`
- Modify: \`modules/translators/trans_llm_api_json.py\`

**Interfaces:**
- Consumes Task 1 helpers.
- Produces \`api_key_tier -> str\` and \`active_api_keys -> List[str]\`.
- Preserves \`_select_api_key() -> Optional[str]\` and per-key limits.

- [ ] **Step 1: Write failing translator tests**

\`\`\`python
class LLMTranslatorKeyPoolTest(unittest.TestCase):
    def make_translator(self, **params):
        return OpenAILLMTranslator(
            "日本語", "한국어", raise_unsupported_lang=False,
            **{
                "api_key_tier": "Free",
                "free_api_keys": "free-a;free-b",
                "paid_api_keys": "paid-a;paid-b",
                "max requests per minute": 0,
                **params,
            },
        )

    def test_rotates_only_selected_pool_and_resets_on_switch(self):
        translator = self.make_translator()
        self.assertEqual(
            [translator._select_api_key() for _ in range(3)],
            ["free-a", "free-b", "free-a"],
        )
        translator.updateParam("api_key_tier", "Paid")
        self.assertEqual(translator.current_key_index, 0)
        self.assertIsNone(translator.client)
        self.assertEqual(
            [translator._select_api_key() for _ in range(3)],
            ["paid-a", "paid-b", "paid-a"],
        )

    def test_single_paid_key_uses_the_pool_path(self):
        translator = self.make_translator(
            api_key_tier="Paid", paid_api_keys="paid-only"
        )
        self.assertEqual(translator._select_api_key(), "paid-only")
        self.assertEqual(translator._select_api_key(), "paid-only")
\`\`\`

Also assert fixed-provider schemas have a Free-default selector and two editors, and an empty Paid pool returns \`None\` even if Free keys exist.

- [ ] **Step 2: Run tests and verify RED**

Run: \`.venv/bin/python -m unittest tests.test_llm_api_key_pools.LLMTranslatorKeyPoolTest\`

Expected: schema/property failures because selection still uses legacy fields.

- [ ] **Step 3: Implement translator behavior**

Replace visible legacy fields with \`api_key_tier\`, \`free_api_keys\`, and \`paid_api_keys\`. Resolve a non-empty editor first, otherwise call \`get_llm_api_key_pool(provider, tier)\`. Keep \`apikey\` and \`multiple_keys_list\` as compatibility views of the active pool. Rotate every pool, including a single-key Paid pool, through the same modulo path. Tier or pool changes reset \`client\` and \`current_key_index\`.

- [ ] **Step 4: Run translator regressions**

Run: \`.venv/bin/python -m unittest tests.test_llm_api_key_pools.LLMTranslatorKeyPoolTest tests.test_local_translators tests.test_gemini_reasoning\`

Expected: all tests pass.

- [ ] **Step 5: Commit**

\`\`\`bash
git add modules/translators/trans_llm_api_json.py tests/test_llm_api_key_pools.py
git commit -m "feat: select translator API key tiers"
\`\`\`

---

### Task 3: OCR tier selection

**Files:**
- Modify: \`tests/test_llm_api_key_pools.py\`
- Modify: \`modules/ocr/ocr_llm_api.py\`

**Interfaces:**
- Consumes Task 1 helpers with \`for_ocr=True\`.
- Produces \`api_key_tier -> str\` and \`active_api_keys -> List[str]\`.
- Preserves local-provider dummy-key handling in \`ocr()\`.

- [ ] **Step 1: Write failing OCR tests**

\`\`\`python
class LLMOCRKeyPoolTest(unittest.TestCase):
    def make_ocr(self, **params):
        return OpenAILLMOCR(**{
            "api_key_tier": "Free",
            "free_api_keys": "ocr-free-a;ocr-free-b",
            "paid_api_keys": "ocr-paid-a;ocr-paid-b",
            "requests_per_minute": 0,
            **params,
        })

    def test_rotates_only_selected_pool_and_resets_on_switch(self):
        ocr = self.make_ocr()
        self.assertEqual(
            [ocr._select_api_key() for _ in range(3)],
            ["ocr-free-a", "ocr-free-b", "ocr-free-a"],
        )
        ocr.updateParam("api_key_tier", "Paid")
        self.assertEqual(ocr.current_key_index, 0)
        self.assertIsNone(ocr.client)
        self.assertEqual(
            [ocr._select_api_key() for _ in range(2)],
            ["ocr-paid-a", "ocr-paid-b"],
        )
\`\`\`

Add an environment-backed namespace test proving OCR cannot read translator keys. Assert an empty selected Paid pool does not consume configured Free keys.

- [ ] **Step 2: Run tests and verify RED**

Run: \`.venv/bin/python -m unittest tests.test_llm_api_key_pools.LLMOCRKeyPoolTest\`

Expected: failures because OCR still uses legacy key fields.

- [ ] **Step 3: Implement OCR behavior**

Mirror the translator schema and selection properties, using OCR environment variables. Change \`_select_api_key\` to rotate only over \`active_api_keys\`. Tier/pool changes reset \`client\` and \`current_key_index\`; retain existing delay and RPM resets.

- [ ] **Step 4: Run combined tests**

Run: \`.venv/bin/python -m unittest tests.test_llm_api_key_pools tests.test_llm_env tests.test_gemini_reasoning\`

Expected: all tests pass.

- [ ] **Step 5: Commit**

\`\`\`bash
git add modules/ocr/ocr_llm_api.py tests/test_llm_api_key_pools.py
git commit -m "feat: select OCR API key tiers"
\`\`\`

---

### Task 4: Migration and full verification

**Files:**
- Modify: \`tests/test_llm_env.py\`
- Modify only if a failing migration test requires it: \`utils/config.py\`

**Interfaces:**
- Verifies old config secrets persist to Free-tier environment variables before obsolete parameters are patched out.
- Verifies new tier settings survive \`ProgramConfig.load\`.

- [ ] **Step 1: Write the end-to-end migration test**

Load a temporary legacy config and persist its module settings to a temporary \`.env\`. Assert:

\`\`\`python
self.assertEqual(
    dotenv_values["BALLOONTRANS_LLM_OPENAI_FREE_API_KEYS"],
    "legacy-single;legacy-a;legacy-b",
)
self.assertEqual(
    dotenv_values["BALLOONTRANS_LLM_OCR_GOOGLE_FREE_API_KEYS"],
    "legacy-ocr;legacy-ocr-b",
)
self.assertEqual(
    config.module.translator_params["LLM OpenAI"]["api_key_tier"], "Free"
)
self.assertEqual(
    config.module.ocr_params["LLM OCR Google"]["api_key_tier"], "Free"
)
\`\`\`

- [ ] **Step 2: Run migration tests**

Run: \`.venv/bin/python -m unittest tests.test_llm_env\`

Expected: PASS if Tasks 1-3 cover migration; otherwise the focused assertion identifies the missing hook.

- [ ] **Step 3: Add the minimal migration hook only if RED**

If required, add \`_migrate_llm_api_key_params(module_cfg)\` in \`utils/config.py\` before parameter patching. It sets a missing tier to Free, combines legacy secrets into \`free_api_keys\`, initializes \`paid_api_keys\` empty, and delegates parsing/naming to \`utils.env\`.

- [ ] **Step 4: Run final verification**

\`\`\`bash
.venv/bin/python -m unittest tests.test_llm_api_key_pools tests.test_llm_env tests.test_local_translators tests.test_gemini_reasoning
.venv/bin/python -m py_compile utils/env.py modules/translators/trans_llm_api_json.py modules/ocr/ocr_llm_api.py tests/test_llm_env.py tests/test_llm_api_key_pools.py
git diff --check
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
\`\`\`

Expected: focused and full suites pass, compilation succeeds, and diff check is clean.

- [ ] **Step 5: Commit migration coverage**

\`\`\`bash
git add utils/config.py tests/test_llm_env.py
git commit -m "test: cover API key tier migration"
\`\`\`

Skip unchanged files when staging. Never stage \`config/textstyles/default.json\`.
