# Free/Paid API Key Pools Design

## Goal

Allow every remote LLM translator and LLM OCR module to select either a Free or Paid API-key pool. Translation and OCR make their selections independently, and requests rotate only within the selected pool.

## User-facing behavior

Each applicable translator and OCR settings panel exposes:

- `API key tier`: `Free` or `Paid`, defaulting to `Free`.
- `Free API keys`: one or more keys separated by semicolons or newlines.
- `Paid API keys`: one or more keys separated by semicolons or newlines.

The selected pool is the only pool used for requests. Keys in that pool are used in round-robin order with the existing per-key request-limit tracking. The application never falls back from Free to Paid or from Paid to Free automatically. If the selected pool is empty or every key in it is unavailable, the module returns its existing no-key/rate-limit error.

Local providers such as LLM Studio and Ollama continue to use their existing dummy-key behavior and do not require a configured pool.

## Compatibility and migration

The existing single-key and multiple-key settings and environment variables are treated as legacy Free keys. On upgrade:

- Existing keys remain usable without user action.
- The default selected tier is Free.
- A legacy single key and legacy multiple-key list are combined in order, with duplicates removed, when resolving the Free pool.
- New tier-aware values take precedence over legacy values when present.

This compatibility path avoids deleting legacy environment variables. New saves use the tier-aware environment variables, while legacy variables remain readable for older installations and manual `.env` configurations.

## Secret storage

API keys continue to be persisted in `.env` rather than ordinary configuration JSON. Tier-aware variables follow the existing provider-specific split:

- Translator: `BALLOONTRANS_LLM_<PROVIDER>_FREE_API_KEYS` and `BALLOONTRANS_LLM_<PROVIDER>_PAID_API_KEYS`
- OCR: `BALLOONTRANS_LLM_OCR_<PROVIDER>_FREE_API_KEYS` and `BALLOONTRANS_LLM_OCR_<PROVIDER>_PAID_API_KEYS`

The selected tier is not secret and remains in normal module configuration. Saved configuration JSON sanitization removes both Free and Paid key values.

## Components and data flow

1. The parameter schemas in the shared LLM translator and OCR implementations define the tier selector and two key-list editors.
2. The generic parameter UI renders these fields using its existing selector and multiline-editor controls.
3. Environment helpers persist, sanitize, and resolve tier-aware key pools. Free resolution falls back to legacy single/multiple variables for migration.
4. Translator and OCR key-selection methods request the active pool, maintain an independent round-robin index, and apply existing per-key request limits.
5. Changing a tier or either pool resets the client and round-robin index so the following request uses the newly selected configuration.

Fixed-provider LLM modules inherit the same behavior from their shared base implementation. Translation and OCR use distinct environment-variable namespaces and therefore cannot consume each other's keys accidentally.

## Error handling

- Empty entries and surrounding whitespace are ignored.
- Duplicate keys within the resolved pool are removed while preserving first occurrence order.
- An empty selected remote pool produces the current explicit no-key failure.
- Exhausting the selected pool produces the current rate-limit failure; no cross-tier fallback occurs.
- Unknown or missing tier values normalize to `Free` for compatibility.

## Testing

Automated tests cover:

- Free and Paid pools are parsed independently.
- Round-robin selection stays within the active tier for translators and OCR.
- Switching tiers resets selection state and client state.
- A single Paid key works through the same pool mechanism.
- Legacy single and multiple keys resolve as the Free pool by default.
- Tier-aware environment variables are persisted and secret values are removed from saved JSON.
- Translation and OCR environment namespaces remain independent.
- Existing LLM and full regression suites continue to pass.
