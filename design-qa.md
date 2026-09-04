# Compact settings UI — 2026-09-04

final result: passed

## Visual target and evidence

- Approved source: `doc/src/settings-panel-approved.png` (the combined design after removing the model-row highlight).
- Implementation: `doc/src/settings-panel-compact.png`.
- Additional captures: `doc/src/settings-panel-dark.png`, `doc/src/settings-panel-ocr.png`.
- Target state: Korean, light theme, Text Detection selected, CTD, 1280 / 4 / mps / 1.0 / -1 / -1 / 2, keep-existing-lines off.
- Source pixels: 1402 × 1122 including generated window chrome. Implementation: 900 × 700 client area, device pixel ratio 1. This is a native Qt application, not a browser prototype.
- Density comparison: source content below its approximately 65px title bar corresponds to approximately 900 × 679 after uniform scaling to 900px width. Compare client regions and proportions; do not count the absent native title bar in the offscreen capture as design drift. No source artwork was stretched or used as the running interface.
- Full-view evidence: source and implementation were opened together in the same multi-image comparison input after each visual pass. Final comparison also included the long OCR form and dark theme. These were paired image views, not a claimed pixel-diff or a side-by-side composite.
- Focused areas: labels, field edges, selection treatment, and icons were clearly readable in those full-resolution inputs; an additional crop was not needed.

## Comparison history

1. First render: P2 — 14px form text was smaller and rows denser than the approved target; mixed legacy icons did not have the target's consistent line style. The DL overview also crowded action buttons into the cache-setting row.
2. Fixes: 16px form type, 50px parameter-row rhythm, larger section gaps, consistent bundled Tabler outline icons, and separate overview action rows.
3. Second render: labels, values, and navigation fit. Added a fixed page heading outside the scroll area and verified it stays visible on long pages. Final render retained the corrected geometry; no actionable P0/P1/P2 differences remain within the native-control constraints below.
4. Code review found shared inpaint-selector reparenting could remove the canvas control while modeless settings were open. Settings and drawing tools now retain separate selectors with synchronized values. Compact multi-checkbox parameters now stack vertically to keep labels readable at 800px. Added regressions for both findings; updated the paint-test fixture to provide the real combobox interface.

## Required fidelity surfaces

- Typography: Apple SD Gothic Neo with platform fallback; 22px page title, 17px section headings, 16px form content. Long names wrap rather than making the page wider. Labels and numeric values are legible.
- Layout: 230px fixed sidebar, 28px content inset, consistently right-aligned 200px simple controls. Large prompt editors use the available content width. The CTD page fits without either scrollbar at 900 × 700. Long pages use only vertical scrolling, with a fixed title and navigation.
- Color: white content, very light neutral sidebar, pale sky-blue selected navigation (`#e0f2fe`) and sky-blue leading indicator (`#38bdf8`). Model selection has no colored band, hover background, or outer card. Form focus and switch states use neutral colors.
- Assets: genuine bundled MIT-licensed Tabler SVGs, consistently recolored at render time. No generated screenshot is used to simulate controls. The exact icon silhouettes differ slightly from the image-generated symbols but retain their meaning and consistent stroke style.
- Copy: original navigation hierarchy and controls remain. Korean aliases for the displayed detector parameters do not change stored parameter keys. Existing untranslated module-specific copy remains unchanged.

## Interaction and regression evidence

- 16 floating-settings tests cover modeless window behavior, canvas retention, title/localization, centering, close/launcher synchronization, existing hierarchy, selected-page navigation, no horizontal overflow at 800/900px, neutral hover, retained prompt edits, fixed heading while scrolling, keyboard toggle persistence, parameter-edit signal payloads, language option loading, synchronized independent canvas selectors, and fully visible compact checkbox labels.
- All 42 registered module forms were constructed from actual registry metadata and switched through: 6 detectors, 21 OCR engines, 6 inpainters, 9 translators. Each had a horizontal scroll range of zero at 900 × 700. No model download or inference was run for this check.
- Full unittest suite: 275 tests passed (`python -m unittest discover -s tests -q`, fresh final run).
- Python compilation and `git diff --check` passed.

## Native-platform constraints and follow-up polish

- Captures and interaction checks use Qt's offscreen backend. Live macOS window-server chrome, multi-monitor placement, and full running-editor integration were not visually re-tested in this pass. Native title-bar traffic lights are supplied by macOS, not drawn by the application.
- P3: numeric values retain the existing validated line editors rather than adding spin-step semantics inferred only from generated artwork. Tabler icons are close library equivalents, not tracings of image-generated icons.
- P3: exact native popup/arrow rendering depends on platform theme. Dark-mode controls were captured for geometry and foreground/background review; existing application theme handling continues to own shared popup-arrow assets.

## Implementation checklist

- [x] Preserve settings hierarchy and existing signal/storage contracts.
- [x] Apply approved neutral model row and sky-blue navigation accent.
- [x] Keep controls inside the compact window without horizontal scrolling.
- [x] Inspect actual Qt renders and run regression tests.
- [x] Leave application restart to the user; do not replace their running session or commit unrelated work.
