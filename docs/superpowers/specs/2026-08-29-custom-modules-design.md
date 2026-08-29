# Custom Modules Design

## Goal

Add an update-safe `custom_modules/` extension point so users can install translators, OCR engines, text detectors, and inpainters without editing SekaiTranslator2's built-in `modules/` tree.

## Source and Reference

This design implements PR 1 from the ChatGPT comparison conversation `비교 이식 요소 정리` (`6a91a1db-83f4-83e8-9719-df72e7b51c34`). The behavior is adapted from `SangGuKim/BallonsTranslator` main at `0ca496533763e6945e8280989b174be1802fbded`; only its user-module discovery, error isolation, and updater-preservation ideas are in scope.

SekaiTranslator2 keeps its flat package layout, current AST-based lazy registry, provider/API-key extensions, headless server, relay server, and local worker unchanged.

## Supported Module Contract

Custom module files live directly under `<program root>/custom_modules/`. Discovery is non-recursive and uses the same filename and decorator contracts as built-in modules:

| Module type | Filename | Decorator |
|---|---|---|
| Translator | `trans_*.py` | `@register_translator("name")` |
| OCR | `ocr_*.py` | `@register_OCR("name")` |
| Text detector | `detector_*.py` | `@register_textdetectors("name")` |
| Inpainter | `inpaint_*.py` | `@register_inpainter("name")` |

The directory is rooted at `Path(shared.PROGRAM_PATH) / "custom_modules"`, not at the process working directory. Custom files use the import namespace `custom_modules.<filename-without-.py>`.

The normal startup path remains lazy: source files are parsed into `ModuleSpec` records without executing their top-level code, and the selected module is imported only when its registry entry is resolved. `import_module_registries()` remains available as the explicit eager compatibility/debug path and searches the same custom directory after built-ins.

Adding, removing, or editing a custom module requires an application restart. Runtime reload and registry invalidation are not part of PR 1.

## Ordering and Collision Policy

Built-in files are scanned first. Custom files are scanned afterward in sorted filename order.

A custom module cannot replace an existing built-in registry key. It also cannot replace a key registered by an earlier custom file. The first registered key remains active, and the conflicting custom file produces a warning. This protects built-in behavior and makes duplicate resolution deterministic.

## Error Handling

Failure in one custom file must not abort registry initialization or hide valid sibling modules. Syntax errors, unreadable files, metadata-scan failures, duplicate keys, and eager-import failures are logged with the file path and then skipped.

Failures in built-in module scanning continue to raise. Silently swallowing built-in errors would make application regressions harder to detect.

A custom file whose AST metadata is valid but whose runtime dependency is missing remains visible through its lazy spec. Selecting it uses the existing `ModuleSpec.resolve()` behavior and raises a module-specific `LazyModuleError`; other registries remain usable.

## Update Safety

`custom_modules/*` is ignored by Git so `git pull`, `git stash -u`, and ordinary source updates do not treat installed custom modules as repository changes. A tracked `custom_modules/put your custom modules here.txt` marker keeps the directory in clean checkouts and points users to documentation.

The source-zip updater continues to replace only paths listed in `SOURCE_UPDATE_DIRS` and `SOURCE_UPDATE_FILES`. `custom_modules` must remain absent from both allowlists, and a regression test locks down that contract.

## Security

Custom modules are trusted local Python code. Lazy discovery does not execute their top-level code, but selecting a module imports it with the same permissions as SekaiTranslator2. Documentation must tell users to install custom files only from sources they trust. PR 1 does not attempt process isolation, permission restriction, signature verification, or dependency sandboxing.

## Documentation

Create one English guide covering supported filenames, a minimal translator example, restart behavior, collision/error behavior, update preservation, dependency declarations, and the trusted-code warning. Link it from `README_EN.md` and the existing translator-extension guide.

## Testing

Tests must verify:

1. the custom root is derived from `shared.PROGRAM_PATH`;
2. all four filename patterns are discovered non-recursively;
3. lazy discovery creates an unresolved `ModuleSpec` without executing top-level code;
4. resolving a valid custom spec imports its class;
5. a malformed custom file is warned and skipped while a valid sibling remains available;
6. a duplicate custom key does not override the existing registry entry;
7. the eager compatibility loader imports a valid custom module and isolates a failing sibling;
8. an absent custom directory is harmless;
9. custom files are ignored by Git and excluded from updater-managed paths;
10. the existing lazy-runtime and updater suites remain green.

## Non-Goals

- Copying BallonsTranslator's package namespace or entry-point layout.
- Installing, enabling, disabling, updating, or deleting custom modules from the GUI.
- Recursive packages or one subdirectory per module.
- Hot reload without restarting the application.
- Allowing custom modules to override built-in registry keys.
- Automatically installing arbitrary dependencies during discovery.
- Sandboxing or signing third-party code.

