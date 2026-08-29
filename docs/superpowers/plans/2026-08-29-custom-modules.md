# Custom Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an update-safe `custom_modules/` directory whose translator, OCR, text-detector, and inpainter files participate in SekaiTranslator2's existing eager and AST-based lazy registries without endangering built-in module startup.

**Architecture:** `modules.base` owns the program-root custom path and mirrors built-in discovery in the eager compatibility path. `modules.lazy_registry` appends custom candidates to the existing built-in scan, maps them to the `custom_modules.*` namespace, preserves lazy import behavior, and catches errors only for user files. Git ignore rules and the updater allowlist keep user files outside managed source updates.

**Tech Stack:** Python 3, `pathlib`, `importlib`, AST metadata scanning, existing `Registry`/`ModuleSpec`, `unittest`, Git.

**Spec:** `docs/superpowers/specs/2026-08-29-custom-modules-design.md`

## Global Constraints

- Preserve SekaiTranslator2's flat package layout; do not introduce the `ballontranslator/` namespace.
- Support exactly the existing `translator`, `ocr`, `textdetector`, and `inpainter` registry groups.
- Discover only top-level files matching `trans_*.py`, `ocr_*.py`, `detector_*.py`, or `inpaint_*.py`.
- Keep built-in modules ahead of custom modules; custom files cannot override an existing registry key.
- Keep normal startup lazy and free of custom top-level code execution.
- Catch and log custom-file failures, but continue raising built-in scan failures.
- Require an application restart after custom files change; do not add hot reload.
- Do not auto-install arbitrary custom dependencies during discovery.
- Treat custom files as trusted code and document that they execute with application permissions when selected.
- Keep `custom_modules` outside `SOURCE_UPDATE_DIRS` and `SOURCE_UPDATE_FILES`.
- Preserve unrelated working-tree changes, especially `config/textstyles/default.json`.
- Add no runtime dependency.

---

### Task 1: Establish the user-owned directory and update-safety contract

**Files:**
- Create: `custom_modules/put your custom modules here.txt`
- Modify: `.gitignore:1-18`
- Modify: `tests/test_updater.py:14-28`

**Interfaces:**
- Produces: `<program root>/custom_modules/` as a present but user-owned directory.
- Produces: Git rules that ignore every installed custom file while tracking the directory marker.
- Preserves: `utils.updater.SOURCE_UPDATE_DIRS` and `SOURCE_UPDATE_FILES` as explicit managed-path allowlists that exclude `custom_modules`.

- [ ] **Step 1: Write the failing ownership-contract test**

Add the import at the top of `tests/test_updater.py`, then add the method to `UpdaterTest`:

```python
from utils import updater as updater_module


def test_custom_modules_are_user_owned_and_ignored(self):
    project_root = Path(__file__).resolve().parents[1]
    ignore_lines = {
        line.strip()
        for line in (project_root / ".gitignore").read_text(encoding="utf8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    self.assertNotIn("custom_modules", updater_module.SOURCE_UPDATE_DIRS)
    self.assertNotIn("custom_modules", updater_module.SOURCE_UPDATE_FILES)
    self.assertIn("custom_modules/*", ignore_lines)
    self.assertIn("!custom_modules/put your custom modules here.txt", ignore_lines)
    self.assertTrue((project_root / "custom_modules" / "put your custom modules here.txt").is_file())
```

- [ ] **Step 2: Run the ownership test and verify RED**

Run: `.venv/bin/python -m unittest tests.test_updater.UpdaterTest.test_custom_modules_are_user_owned_and_ignored -v`

Expected: FAIL because the ignore rules and tracked marker do not exist.

- [ ] **Step 3: Add the ignore rules and tracked marker**

Append to `.gitignore`:

```gitignore
# User-installed modules are preserved across source updates.
custom_modules/*
!custom_modules/put your custom modules here.txt
```

Create `custom_modules/put your custom modules here.txt` with:

```text
Place trusted custom module files directly in this directory.

Supported filenames:
- trans_*.py for translators
- ocr_*.py for OCR engines
- detector_*.py for text detectors
- inpaint_*.py for inpainters

Restart SekaiTranslator2 after adding or changing a file.
See doc/custom_modules.md for the module contract, examples, and security notes.
```

Do not add `custom_modules` to `utils/updater.py`; its absence from the updater allowlists is the preservation mechanism.

- [ ] **Step 4: Run the ownership test and Git-ignore check**

Run: `.venv/bin/python -m unittest tests.test_updater.UpdaterTest.test_custom_modules_are_user_owned_and_ignored -v`

Expected: PASS.

Run: `git check-ignore -q custom_modules/trans_user_example.py`

Expected: exit code 0.

- [ ] **Step 5: Commit the user-owned directory contract**

```bash
git add .gitignore "custom_modules/put your custom modules here.txt" tests/test_updater.py
git commit -m "feat: reserve update-safe custom module directory"
```

---

### Task 2: Add custom discovery to the eager compatibility loader

**Files:**
- Create: `tests/test_custom_modules.py`
- Modify: `modules/base.py:512-555`

**Interfaces:**
- Produces: `CUSTOM_MODULE_ROOT: Path = Path(shared.PROGRAM_PATH) / "custom_modules"`.
- Extends: `import_module_registries(target_modules=None) -> None` to import matching built-in files first and matching `custom_modules.*` files second.
- Preserves: one-file failure isolation through a warning and continuation.

- [ ] **Step 1: Write the failing eager-discovery test**

Create `tests/test_custom_modules.py`:

```python
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CustomModuleEagerDiscoveryTest(unittest.TestCase):
    def test_eager_loader_imports_valid_custom_module_after_bad_sibling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            custom_root = temp_root / "custom_modules"
            builtin_root = temp_root / "builtin_translators"
            custom_root.mkdir()
            builtin_root.mkdir()
            (custom_root / "trans_00_broken.py").write_text(
                'raise RuntimeError("broken custom module")\n',
                encoding="utf8",
            )
            (custom_root / "trans_10_echo.py").write_text(
                textwrap.dedent(
                    """
                    from modules.translators.base import BaseTranslator, register_translator

                    @register_translator("Custom Echo Eager")
                    class CustomEchoEager(BaseTranslator):
                        def _setup_translator(self):
                            self.lang_map["日本語"] = "ja"
                            self.lang_map["English"] = "en"

                        def _translate(self, src_list):
                            return list(src_list)
                    """
                ),
                encoding="utf8",
            )

            script = textwrap.dedent(
                """
                import sys
                from pathlib import Path
                from unittest.mock import patch

                import modules
                from modules import base

                custom_root = Path(sys.argv[1])
                builtin_root = Path(sys.argv[2])
                sys.path.insert(0, str(custom_root.parent))
                translator_script = dict(base.MODULE_SCRIPTS["translator"])
                translator_script["module_dir"] = str(builtin_root)

                with patch.object(base, "CUSTOM_MODULE_ROOT", custom_root), patch.dict(
                    base.MODULE_SCRIPTS,
                    {"translator": translator_script},
                    clear=False,
                ):
                    base.import_module_registries("translator")

                assert "Custom Echo Eager" in modules.TRANSLATORS.module_dict
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(custom_root), str(builtin_root)],
                cwd=str(PROJECT_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_root_uses_program_path(self):
        from modules.base import CUSTOM_MODULE_ROOT
        from utils import shared

        self.assertEqual(CUSTOM_MODULE_ROOT, Path(shared.PROGRAM_PATH) / "custom_modules")

    def test_missing_custom_directory_is_harmless(self):
        from modules import base

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            builtin_root = temp_root / "builtin_translators"
            builtin_root.mkdir()
            translator_script = dict(base.MODULE_SCRIPTS["translator"])
            translator_script["module_dir"] = str(builtin_root)

            with patch.object(
                base,
                "CUSTOM_MODULE_ROOT",
                temp_root / "missing",
            ), patch.dict(
                base.MODULE_SCRIPTS,
                {"translator": translator_script},
                clear=False,
            ):
                base.import_module_registries("translator")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the eager tests and verify RED**

Run: `.venv/bin/python -m unittest tests.test_custom_modules.CustomModuleEagerDiscoveryTest -v`

Expected: ERROR because `modules.base.CUSTOM_MODULE_ROOT` does not exist.

- [ ] **Step 3: Implement deterministic eager custom discovery**

In `modules/base.py`, define the root next to `MODULE_ROOT`:

```python
MODULE_ROOT = Path(__file__).resolve().parent
CUSTOM_MODULE_ROOT = Path(shared.PROGRAM_PATH) / "custom_modules"
```

Replace the nested eager helper and loop with:

```python
def import_module_registries(target_modules=None):
    def _load_module(module_dir: str, module_package: str, module_pattern: str):
        if not os.path.isdir(module_dir):
            return
        pattern = re.compile(module_pattern)
        for module_name in sorted(os.listdir(module_dir)):
            if pattern.match(module_name) is None:
                continue
            module = module_package + "." + module_name.replace(".py", "")
            try:
                importlib.import_module(module)
            except Exception as e:
                LOGGER.warning(f"Failed to import {module}: {e}")

    if target_modules is None:
        target_modules = MODULE_SCRIPTS
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    for module_type in target_modules:
        script = MODULE_SCRIPTS[module_type]
        _load_module(**script)
        _load_module(
            str(CUSTOM_MODULE_ROOT),
            "custom_modules",
            script["module_pattern"],
        )
```

The directory guard makes a missing custom directory a no-op. Sorting makes failure and collision order reproducible.

- [ ] **Step 4: Run the eager tests and verify GREEN**

Run: `.venv/bin/python -m unittest tests.test_custom_modules.CustomModuleEagerDiscoveryTest -v`

Expected: PASS with 3 tests; the missing directory is harmless, the broken file emits a warning, and `Custom Echo Eager` is still registered.

- [ ] **Step 5: Commit eager discovery**

```bash
git add modules/base.py tests/test_custom_modules.py
git commit -m "feat: discover eager custom modules"
```

---

### Task 3: Integrate custom files into lazy discovery with isolation and precedence

**Files:**
- Modify: `modules/lazy_registry.py:14-46`
- Modify: `modules/lazy_registry.py:542-550`
- Modify: `modules/lazy_registry.py:855-884`
- Modify: `tests/test_custom_modules.py`

**Interfaces:**
- Produces: `_module_files(module_type: str) -> List[str]`, returning built-in candidates, extra base files, then custom candidates.
- Produces: `_is_custom_module_file(path: str) -> bool`.
- Extends: `_module_name_from_path(path: str) -> str` with the `custom_modules.<stem>` namespace.
- Extends: `init_lazy_module_registries(target_modules=None) -> None` so custom scan/registration failures are logged and skipped while built-in failures still raise.
- Consumes: `CUSTOM_MODULE_ROOT` from Task 2.
- Preserves: existing `ModuleSpec` fields, `INITIALIZED_REGISTRIES` idempotence, built-in metadata evaluation, and lazy runtime imports.

- [ ] **Step 1: Write failing lazy-discovery tests**

Add a helper and test class:

```python
class CustomModuleLazyDiscoveryTest(unittest.TestCase):
    def _run_script(self, script, *args):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script), *map(str, args)],
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_custom_filename_patterns_are_collected_nonrecursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_root = Path(temp_dir) / "custom_modules"
            custom_root.mkdir()
            for filename in (
                "trans_demo.py",
                "ocr_demo.py",
                "detector_demo.py",
                "inpaint_demo.py",
                "ignored.py",
            ):
                (custom_root / filename).write_text("", encoding="utf8")
            nested = custom_root / "nested"
            nested.mkdir()
            (nested / "trans_nested.py").write_text("", encoding="utf8")

            self._run_script(
                """
                import sys
                from pathlib import Path
                from unittest.mock import patch

                from modules import lazy_registry

                custom_root = Path(sys.argv[1])
                empty_root = Path(sys.argv[2])
                empty_root.mkdir()
                scripts = {
                    key: {**value, "module_dir": str(empty_root / key)}
                    for key, value in lazy_registry.MODULE_SCRIPTS.items()
                }
                for value in scripts.values():
                    Path(value["module_dir"]).mkdir()
                with patch.object(lazy_registry, "CUSTOM_MODULE_ROOT", custom_root), patch.dict(
                    lazy_registry.MODULE_SCRIPTS,
                    scripts,
                    clear=True,
                ), patch.object(lazy_registry, "EXTRA_MODULE_FILES", {}):
                    names = {
                        key: [Path(path).name for path in lazy_registry._module_files(key)]
                        for key in scripts
                    }
                assert names == {
                    "translator": ["trans_demo.py"],
                    "textdetector": ["detector_demo.py"],
                    "inpainter": ["inpaint_demo.py"],
                    "ocr": ["ocr_demo.py"],
                }, names
                """,
                custom_root,
                Path(temp_dir) / "builtins",
            )

    def test_lazy_scan_defers_import_and_skips_malformed_sibling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            custom_root = temp_root / "custom_modules"
            builtin_root = temp_root / "builtins"
            custom_root.mkdir()
            builtin_root.mkdir()
            marker = temp_root / "custom_loaded.txt"
            (custom_root / "trans_00_broken.py").write_text(
                "def invalid(:\n",
                encoding="utf8",
            )
            (custom_root / "trans_10_echo.py").write_text(
                textwrap.dedent(
                    f"""
                    from pathlib import Path
                    Path({str(marker)!r}).write_text("loaded", encoding="utf8")
                    from modules.translators.base import BaseTranslator, register_translator

                    @register_translator("Custom Echo Lazy")
                    class CustomEchoLazy(BaseTranslator):
                        params = {{"prefix": {{"value": "[custom]"}}}}

                        @property
                        def supported_src_list(self):
                            return ["日本語"]

                        @property
                        def supported_tgt_list(self):
                            return ["English"]

                        def _setup_translator(self):
                            self.lang_map["日本語"] = "ja"
                            self.lang_map["English"] = "en"

                        def _translate(self, src_list):
                            return list(src_list)
                    """
                ),
                encoding="utf8",
            )

            self._run_script(
                """
                import sys
                from pathlib import Path
                from unittest.mock import patch

                import modules
                from modules import lazy_registry
                from utils.registry import ModuleSpec, Registry

                custom_root = Path(sys.argv[1])
                builtin_root = Path(sys.argv[2])
                marker = Path(sys.argv[3])
                sys.path.insert(0, str(custom_root.parent))
                registry = Registry("custom-translators")
                script = dict(lazy_registry.MODULE_SCRIPTS["translator"])
                script["module_dir"] = str(builtin_root)
                lazy_registry.INITIALIZED_REGISTRIES.discard("translator")

                with patch.object(lazy_registry, "CUSTOM_MODULE_ROOT", custom_root), patch.dict(
                    lazy_registry.MODULE_SCRIPTS,
                    {"translator": script},
                    clear=False,
                ), patch.dict(
                    modules.MODULETYPE_TO_REGISTRIES,
                    {"translator": registry},
                    clear=False,
                ), patch.object(lazy_registry, "EXTRA_MODULE_FILES", {}), patch.object(
                    lazy_registry.LOGGER,
                    "warning",
                ) as warning:
                    lazy_registry.init_lazy_module_registries("translator")
                    spec = registry.get("Custom Echo Lazy")
                    assert isinstance(spec, ModuleSpec)
                    assert spec.import_path == "custom_modules.trans_10_echo"
                    assert spec.resolved_class is None
                    assert not marker.exists()
                    resolved = registry.resolve_module("Custom Echo Lazy")
                    assert resolved.__name__ == "CustomEchoLazy"
                    assert marker.read_text(encoding="utf8") == "loaded"
                    assert any("trans_00_broken.py" in str(call) for call in warning.call_args_list)
                """,
                custom_root,
                builtin_root,
                marker,
            )

    def test_custom_duplicate_does_not_replace_existing_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            custom_root = temp_root / "custom_modules"
            builtin_root = temp_root / "builtins"
            custom_root.mkdir()
            builtin_root.mkdir()
            (custom_root / "trans_duplicate.py").write_text(
                textwrap.dedent(
                    """
                    from modules.translators.base import register_translator

                    @register_translator("Reserved Key")
                    class CustomDuplicate:
                        pass
                    """
                ),
                encoding="utf8",
            )

            self._run_script(
                """
                import sys
                from pathlib import Path
                from unittest.mock import patch

                import modules
                from modules import lazy_registry
                from utils.registry import Registry

                class BuiltIn:
                    pass

                custom_root = Path(sys.argv[1])
                builtin_root = Path(sys.argv[2])
                registry = Registry("custom-translators")
                registry.register_module(name="Reserved Key", module=BuiltIn)
                script = dict(lazy_registry.MODULE_SCRIPTS["translator"])
                script["module_dir"] = str(builtin_root)
                lazy_registry.INITIALIZED_REGISTRIES.discard("translator")

                with patch.object(lazy_registry, "CUSTOM_MODULE_ROOT", custom_root), patch.dict(
                    lazy_registry.MODULE_SCRIPTS,
                    {"translator": script},
                    clear=False,
                ), patch.dict(
                    modules.MODULETYPE_TO_REGISTRIES,
                    {"translator": registry},
                    clear=False,
                ), patch.object(lazy_registry, "EXTRA_MODULE_FILES", {}), patch.object(
                    lazy_registry.LOGGER,
                    "warning",
                ) as warning:
                    lazy_registry.init_lazy_module_registries("translator")
                    assert registry.get("Reserved Key") is BuiltIn
                    assert warning.called
                """,
                custom_root,
                builtin_root,
            )
```

- [ ] **Step 2: Run the lazy tests and verify RED**

Run: `.venv/bin/python -m unittest tests.test_custom_modules.CustomModuleLazyDiscoveryTest -v`

Expected: FAIL because `_module_files`, `CUSTOM_MODULE_ROOT`, and `LOGGER` are not exposed by `modules.lazy_registry`, and custom files are not scanned.

- [ ] **Step 3: Add custom path mapping and shared candidate discovery**

Change imports at the top of `modules/lazy_registry.py`:

```python
from utils.logger import logger as LOGGER
from utils.registry import ModuleSpec

from .base import CUSTOM_MODULE_ROOT, MODULE_ROOT, MODULE_SCRIPTS
```

Replace `_module_name_from_path` with:

```python
def _module_name_from_path(path: str) -> str:
    path_obj = Path(path).resolve()
    try:
        rel_path = path_obj.relative_to(CUSTOM_MODULE_ROOT.resolve())
        return "custom_modules." + ".".join(rel_path.with_suffix("").parts)
    except ValueError:
        pass
    try:
        rel_path = path_obj.relative_to(PACKAGE_ROOT)
        return ".".join(rel_path.with_suffix("").parts)
    except ValueError:
        module_name = path.replace(os.sep, ".").replace("/", ".")
        return module_name[:-3] if module_name.endswith(".py") else module_name
```

Move candidate collection out of `init_lazy_module_registries` and define:

```python
def _module_files(module_type: str) -> List[str]:
    script = MODULE_SCRIPTS[module_type]
    pattern = re.compile(script["module_pattern"])
    files = []
    module_dir = script["module_dir"]
    if os.path.isdir(module_dir):
        for name in sorted(os.listdir(module_dir)):
            if pattern.match(name):
                files.append(os.path.join(module_dir, name))
    files.extend(EXTRA_MODULE_FILES.get(module_type, []))
    if os.path.isdir(CUSTOM_MODULE_ROOT):
        for name in sorted(os.listdir(CUSTOM_MODULE_ROOT)):
            if pattern.match(name):
                files.append(os.path.join(CUSTOM_MODULE_ROOT, name))
    return [path for path in files if os.path.exists(path)]


def _is_custom_module_file(path: str) -> bool:
    try:
        Path(path).resolve().relative_to(CUSTOM_MODULE_ROOT.resolve())
        return True
    except ValueError:
        return False
```

Built-ins and their extra base files stay first. The custom directory is appended and remains non-recursive.

- [ ] **Step 4: Isolate custom scan and registration failures**

Replace the registration loop in `init_lazy_module_registries` with:

```python
for module_type in targets:
    if module_type in INITIALIZED_REGISTRIES:
        continue
    registry = MODULETYPE_TO_REGISTRIES[module_type]
    for path in _module_files(module_type):
        try:
            for spec in _scan_file(path, module_type):
                registry.register_lazy_module(spec)
                if _is_custom_module_file(path):
                    LOGGER.info(
                        f'Discovered custom {module_type} module "{spec.key}" from {path}'
                    )
        except Exception as e:
            if not _is_custom_module_file(path):
                raise
            LOGGER.warning(f"Failed to register custom module {path}: {e}")
    INITIALIZED_REGISTRIES.add(module_type)
```

Do not call `register_lazy_module(..., force=True)`: duplicate custom keys must not replace built-ins or earlier custom entries.

- [ ] **Step 5: Run focused custom-module tests**

Run: `.venv/bin/python -m unittest tests.test_custom_modules -v`

Expected: PASS with eager discovery, all four filename patterns, lazy deferral/resolution, malformed-file isolation, duplicate precedence, and missing-directory coverage.

- [ ] **Step 6: Run existing registry regression tests**

Run: `.venv/bin/python -m unittest tests.test_lazy_runtime tests.test_local_translators -v`

Expected: PASS; lazy metadata remains complete, `torch` remains unimported during discovery, and existing local translators still register.

- [ ] **Step 7: Commit lazy discovery and isolation**

```bash
git add modules/lazy_registry.py tests/test_custom_modules.py
git commit -m "feat: register custom modules lazily"
```

---

### Task 4: Document installation, trust, and lifecycle behavior

**Files:**
- Create: `doc/custom_modules.md`
- Modify: `README_EN.md:132-169`
- Modify: `doc/how_to_add_new_translator.md:1-12`

**Interfaces:**
- Produces: a user-facing contract for file placement, supported filename/decorator pairs, restart behavior, collision precedence, dependency handling, update preservation, and trusted-code execution.
- Preserves: the existing built-in-module contribution workflow for developers who intend to submit modules upstream.

- [ ] **Step 1: Create the custom-module guide**

Create `doc/custom_modules.md` with this content:

````markdown
# Custom modules

SekaiTranslator2 can load locally installed translators, OCR engines, text detectors, and inpainters without modifying the built-in `modules/` directory.

## Install

1. Copy the Python file directly into `<SekaiTranslator2>/custom_modules/`.
2. Use the filename required by its module type:

   | Type | Filename | Decorator |
   |---|---|---|
   | Translator | `trans_*.py` | `register_translator` |
   | OCR | `ocr_*.py` | `register_OCR` |
   | Text detector | `detector_*.py` | `register_textdetectors` |
   | Inpainter | `inpaint_*.py` | `register_inpainter` |

3. Restart SekaiTranslator2. Custom modules are discovered at startup and imported only when selected.

Files in subdirectories are not discovered. A custom registry name cannot replace a built-in name; rename the custom module's decorator key if the log reports a duplicate.

## Minimal translator

Save this as `custom_modules/trans_echo.py`:

```python
from modules.translators.base import BaseTranslator, register_translator


@register_translator("Custom Echo")
class CustomEchoTranslator(BaseTranslator):
    params = None

    def _setup_translator(self):
        self.lang_map["日本語"] = "ja"
        self.lang_map["English"] = "en"

    def _translate(self, src_list):
        return list(src_list)
```

Use the existing base classes and decorators from `modules.translators.base`, `modules.ocr.base`, `modules.textdetector.base`, or `modules.inpaint.base`. Declare optional Python packages in the class's existing `dependencies` list so the module-preparation flow can report them.

## Errors and updates

One invalid custom file is logged and skipped without disabling built-in modules or valid custom siblings. A module with a missing runtime dependency may still appear in the selector and will show a module-specific import error when selected.

Installed files under `custom_modules/` are ignored by Git and are outside the source updater's managed paths, so application updates preserve them. Back up custom files separately before reinstalling or deleting the application directory.

## Security

Custom modules are Python code and execute with the same permissions as SekaiTranslator2 when selected. Install them only from sources you trust. SekaiTranslator2 does not sandbox, sign, or audit third-party custom modules.
````

- [ ] **Step 2: Link the guide from existing documentation**

Under `# Automation modules` in `README_EN.md`, add:

```markdown
Custom translators, OCR engines, text detectors, and inpainters can be installed without editing the built-in source tree. See [Custom modules](doc/custom_modules.md).
```

Near the introduction of `doc/how_to_add_new_translator.md`, add:

```markdown
For a local translator that should survive application updates, place a `trans_*.py` file in `custom_modules/` and follow [Custom modules](custom_modules.md). Edit `modules/translators/` only when developing a built-in contribution.
```

- [ ] **Step 3: Verify documentation references and tracked/ignored behavior**

Run: `rg -n "custom_modules|Custom modules" README_EN.md doc/custom_modules.md doc/how_to_add_new_translator.md "custom_modules/put your custom modules here.txt"`

Expected: every installation entry point links or points to the same `doc/custom_modules.md` contract.

Run: `git check-ignore -q custom_modules/trans_echo.py`

Expected: exit code 0.

- [ ] **Step 4: Run the complete test suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Run static checks on changed Python files**

Run: `.venv/bin/python -m ruff check modules/base.py modules/lazy_registry.py tests/test_custom_modules.py tests/test_updater.py`

Expected: no lint errors.

- [ ] **Step 6: Review the final diff for scope and user-file safety**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only PR 1 files plus the pre-existing user change `config/textstyles/default.json`; no installed file beneath `custom_modules/` appears except the tracked marker.

- [ ] **Step 7: Commit documentation and verification changes**

```bash
git add README_EN.md doc/custom_modules.md doc/how_to_add_new_translator.md
git commit -m "docs: explain custom module installation"
```

---

## PR Acceptance Checklist

- [ ] A valid custom module appears in the same selector/registry as its built-in peers after restart.
- [ ] Lazy startup does not execute custom top-level code.
- [ ] Selecting a valid custom module resolves its class through `ModuleSpec`.
- [ ] A malformed custom file cannot block a valid sibling or any built-in registry.
- [ ] Duplicate registry keys keep the existing built-in or earlier custom entry.
- [ ] All four supported filename patterns work, and nested files remain ignored.
- [ ] `custom_modules/` stays outside updater-managed source paths and user files are Git-ignored.
- [ ] Existing lazy-runtime, local-translator, updater, and full test suites pass.
- [ ] Documentation warns that custom files are trusted executable Python code.
- [ ] No package namespace migration, GUI manager, hot reload, or automatic third-party dependency installation is included.
