import ast
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {'.git', '.venv', 'data'}
PEP585_BUILTINS = {'list', 'dict', 'tuple', 'set', 'frozenset', 'type'}


def project_python_files():
    for path in PROJECT_ROOT.rglob('*.py'):
        relative_parts = path.relative_to(PROJECT_ROOT).parts
        if any(part in EXCLUDED_PARTS for part in relative_parts):
            continue
        yield path


def module_uses_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == '__future__'
        and any(alias.name == 'annotations' for alias in node.names)
        for node in tree.body
    )


def annotations_in(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            yield node.returns
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation


def annotation_requires_postponed_evaluation(annotation: ast.AST) -> bool:
    for node in ast.walk(annotation):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in PEP585_BUILTINS
        ):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return True
    return False


class Python38CompatibilityTest(unittest.TestCase):
    def test_sources_parse_with_python38_grammar(self):
        failures = []
        for path in project_python_files():
            source = path.read_text(encoding='utf8')
            try:
                feature_version = 8 if sys.version_info[:2] == (3, 8) else (3, 8)
                ast.parse(source, filename=str(path), feature_version=feature_version)
            except SyntaxError as exc:
                failures.append(f'{path.relative_to(PROJECT_ROOT)}:{exc.lineno}: {exc.msg}')
        self.assertEqual(failures, [])

    def test_modern_annotations_are_postponed_for_python38(self):
        failures = []
        for path in project_python_files():
            source = path.read_text(encoding='utf8')
            tree = ast.parse(source, filename=str(path))
            if module_uses_future_annotations(tree):
                continue
            if any(annotation_requires_postponed_evaluation(annotation) for annotation in annotations_in(tree)):
                failures.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(failures, [])


if __name__ == '__main__':
    unittest.main()
