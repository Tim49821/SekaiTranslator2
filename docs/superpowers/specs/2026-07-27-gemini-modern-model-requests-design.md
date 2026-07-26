# Gemini 3.6 Flash and modern request format design

## Goal

Add Gemini 3.6 Flash to both Google LLM translation and Google LLM OCR, and align the request format for the current Gemini models that deprecate sampling parameters.

## Scope

- Add `GGL: gemini-3.6-flash` to the shared Google model catalog.
- Expose the model in both `LLM Google` and `LLM OCR Google`.
- Keep `GGL: gemini-3.1-pro-preview` as the default model.
- Treat `gemini-3.6-flash` and `gemini-3.5-flash-lite` as models that must not receive `temperature` or `top_p`.
- Preserve existing reasoning-effort behavior and structured JSON responses.
- Update lazy metadata, documentation, and automated tests.

Gemini 3.1 models, Gemini 3 Flash Preview, non-Google providers, and unrelated application settings are outside the request-format change.

## Architecture

The Google translator and OCR integrations already consume one shared model catalog from the LLM translator module. Gemini 3.6 Flash will be added to this catalog and to the lazy registry's matching fallback catalog so eager and lazy loading expose identical options.

The translator module will define an explicit set of model IDs that use the modern sampling rules:

- `gemini-3.6-flash`
- `gemini-3.5-flash-lite`

The translation request builder will first resolve and strip the provider prefix from the selected model or override model. It will always send the common request fields. It will add `temperature` and `top_p` only when the resolved model is not in the modern-model set. This also applies the correct behavior to an exact custom override model ID.

The OCR request builder already omits sampling parameters, so it needs no request-format change. It will receive Gemini 3.6 Flash through the shared catalog.

## Data flow

1. The user selects Gemini 3.6 Flash from either Google module's model dropdown.
2. The existing configuration system persists `GGL: gemini-3.6-flash`.
3. The request builder resolves the API model ID to `gemini-3.6-flash`.
4. Translation requests omit `temperature` and `top_p`, while preserving messages, maximum output tokens, JSON response format, and any selected `reasoning_effort`.
5. OCR requests continue to send their existing image, prompt, token-limit, and reasoning-effort fields without sampling parameters.

## Compatibility and error handling

- Existing saved configurations remain valid because the default and existing model IDs do not change.
- Gemini 3.1 and Gemini 3 Flash Preview retain the current translation request fields.
- Only exact current model IDs are classified as modern. Unknown custom models retain the existing request format rather than receiving guessed compatibility behavior.
- API failures continue through the current retry and error-handling paths.

## Testing

Automated tests will verify:

- `GGL: gemini-3.6-flash` appears in the shared translator and OCR Google catalogs.
- Lazy registry metadata contains the same new model.
- Translation requests for Gemini 3.6 Flash omit `temperature` and `top_p`.
- Translation requests for Gemini 3.5 Flash-Lite also omit both sampling parameters.
- A legacy Gemini model retains `temperature` and `top_p`.
- Modern-model requests still send `reasoning_effort` when selected.
- Existing focused and complete test suites continue to pass.

Implementation will follow test-driven development, beginning with failing catalog and request-payload tests.
