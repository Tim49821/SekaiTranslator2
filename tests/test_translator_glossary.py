import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.context.glossary import (
    GLOSSARY_MODE_ALL,
    GLOSSARY_MODE_MATCHING,
    GlossaryEntry,
    GlossaryError,
    load_glossary,
    normalize_glossary_path,
    render_glossary,
    select_glossary,
)


class TranslatorGlossaryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_supported_formats_preserve_file_order(self):
        expected = (
            GlossaryEntry("勇者", "용사", "title"),
            GlossaryEntry("魔王", "마왕", ""),
        )
        cases = {
            "terms.json": (
                '[{"src":"勇者","dst":"용사","info":"title"},'
                '{"src":"魔王","dst":"마왕"}]'
            ),
            "terms.tsv": "勇者\t용사\ttitle\n魔王\t마왕\n",
            "terms.txt": "# comment\n勇者->용사 # title\n魔王->마왕\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                self.assertEqual(load_glossary(self.write(name, content)), expected)

    def test_matching_is_casefolded_literal_and_all_keeps_every_entry(self):
        entries = (
            GlossaryEntry("Hero", "용사"),
            GlossaryEntry("Mage", "마법사"),
        )
        self.assertEqual(
            select_glossary(entries, ["THE HERO arrives"], GLOSSARY_MODE_MATCHING),
            (entries[0],),
        )
        self.assertEqual(select_glossary(entries, ["none"], GLOSSARY_MODE_ALL), entries)

    def test_rendering_is_compact_unicode_json(self):
        self.assertEqual(
            render_glossary((GlossaryEntry("勇者", "용사", "title"),)),
            '{"glossary":[{"source":"勇者","translation":"용사","note":"title"}]}',
        )
        self.assertEqual(render_glossary(()), "")

    def test_normalized_paths_share_cache_and_reload_after_change(self):
        path = self.write("terms.json", '[{"src":"A","dst":"가"}]')
        with patch.dict(os.environ, {"TEST_GLOSSARY_FILE": str(path)}):
            first = load_glossary("$TEST_GLOSSARY_FILE")
            second = load_glossary(path)
        self.assertIs(first, second)
        self.assertEqual(normalize_glossary_path(path), str(path.resolve()))

        path.write_text('[{"src":"A","dst":"나"}]', encoding="utf-8")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        self.assertEqual(load_glossary(path)[0].translation, "나")

    def test_duplicate_rows_collapse_but_conflicting_targets_fail(self):
        duplicate = self.write("duplicate.tsv", "Hero\t용사\nHero\t용사\n")
        self.assertEqual(load_glossary(duplicate), (GlossaryEntry("Hero", "용사"),))

        conflict = self.write("conflict.tsv", "Hero\t용사\nhero\t영웅\n")
        with self.assertRaisesRegex(GlossaryError, "line 2"):
            load_glossary(conflict)

    def test_missing_unsupported_and_malformed_files_report_the_path(self):
        missing = self.root / "missing.json"
        with self.assertRaisesRegex(GlossaryError, "Glossary file not found"):
            load_glossary(missing)

        unsupported = self.write("terms.csv", "Hero,용사\n")
        with self.assertRaisesRegex(GlossaryError, "Unsupported glossary format"):
            load_glossary(unsupported)

        malformed = self.write("bad.json", "{")
        with self.assertRaisesRegex(GlossaryError, "line 1"):
            load_glossary(malformed)

    def test_invalid_mode_fails_concisely(self):
        with self.assertRaisesRegex(GlossaryError, "Invalid glossary mode"):
            select_glossary((GlossaryEntry("A", "가"),), ["A"], "unknown")
