import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ui.module_manager import (
    ImgtransThread,
    InpaintThread,
    OCRThread,
    TextDetectThread,
    TranslateThread,
    translate_project_textblocks,
)
from utils.textblock import TextBlock


class FakeProject:
    def __init__(self, pages):
        self.pages = pages
        self.events = []

    def begin_full_page_translation(self, page_key):
        self.events.append(("begin", page_key))

    def mark_translation_finished(self, page_key, target):
        self.events.append(("finish", page_key, target))


class FakeTranslator:
    lang_target = "한국어"

    def __init__(self, success=True, error=None):
        self.success = success
        self.error = error
        self.calls = []
        self.event_snapshots = []

    def delay(self):
        return 0

    def translate_textblk_lst(self, blocks, **kwargs):
        self.calls.append((blocks, kwargs))
        self.event_snapshots.append(list(kwargs["project"].events))
        if self.error is not None:
            raise self.error
        if self.success:
            for block in blocks:
                if block.get_text().strip():
                    block.translation = "번역"
        return self.success


class LLMContextPipelineTest(unittest.TestCase):
    def test_full_page_invalidates_then_marks_only_after_success(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        translator = FakeTranslator()

        self.assertTrue(
            translate_project_textblocks(
                translator, project, "001.png", blocks, full_page=True
            )
        )

        self.assertEqual(
            project.events,
            [("begin", "001.png"), ("finish", "001.png", "한국어")],
        )
        self.assertEqual(translator.event_snapshots[0], [("begin", "001.png")])
        self.assertEqual(
            translator.calls[0][1],
            {
                "project": project,
                "page_key": "001.png",
                "full_page": True,
            },
        )

    def test_failed_full_page_stays_invalidated(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        translator = FakeTranslator(success=False)

        self.assertFalse(
            translate_project_textblocks(
                translator, project, "001.png", blocks, full_page=True
            )
        )

        self.assertEqual(project.events, [("begin", "001.png")])

    def test_partial_selection_uses_context_without_completing_page(self):
        blocks = [TextBlock(text=["one"]), TextBlock(text=["two"])]
        project = FakeProject({"001.png": blocks})
        translator = FakeTranslator()

        self.assertTrue(
            translate_project_textblocks(
                translator,
                project,
                "001.png",
                blocks[:1],
                full_page=False,
            )
        )

        self.assertEqual(project.events, [])
        self.assertIs(translator.calls[0][1]["project"], project)
        self.assertEqual(translator.calls[0][1]["page_key"], "001.png")
        self.assertFalse(translator.calls[0][1]["full_page"])

    def test_helper_propagates_translation_exceptions(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        error = RuntimeError("translation failed")
        translator = FakeTranslator(error=error)

        with self.assertRaises(RuntimeError) as raised:
            translate_project_textblocks(
                translator, project, "001.png", blocks, full_page=True
            )

        self.assertIs(raised.exception, error)
        self.assertEqual(project.events, [("begin", "001.png")])

    def test_translate_thread_forwards_exact_project_and_page(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        translator = FakeTranslator()
        thread = TranslateThread()
        thread.translator = translator
        thread.module = translator

        success = thread._translate_page(
            project, "001.png", emit_finished=False
        )

        self.assertTrue(success)
        self.assertIs(translator.calls[0][0], blocks)
        self.assertIs(translator.calls[0][1]["project"], project)
        self.assertEqual(translator.calls[0][1]["page_key"], "001.png")

    def test_translate_thread_does_not_swallow_translation_exceptions(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        error = RuntimeError("translation failed")
        translator = FakeTranslator(error=error)
        thread = TranslateThread()
        thread.translator = translator
        thread.module = translator

        with self.assertRaises(RuntimeError) as raised:
            thread._translate_page(
                project, "001.png", emit_finished=False
            )

        self.assertIs(raised.exception, error)

    def test_selected_block_pipeline_forwards_project_and_page_context(self):
        selected = TextBlock(text=["one"])
        unselected = TextBlock(text=["two"])
        project = FakeProject({"001.png": [selected, unselected]})
        translator = FakeTranslator()
        translate_thread = TranslateThread()
        translate_thread.translator = translator
        translate_thread.module = translator
        pipeline = ImgtransThread(
            TextDetectThread(),
            OCRThread(),
            translate_thread,
            InpaintThread(),
        )
        pipeline.imgtrans_proj = project

        pipeline._blktrans_pipeline(
            [selected],
            np.zeros((4, 4, 3), dtype=np.uint8),
            -1,
            [0],
            None,
            page_key="001.png",
        )

        self.assertEqual(project.events, [])
        self.assertIs(translator.calls[0][1]["project"], project)
        self.assertEqual(translator.calls[0][1]["page_key"], "001.png")
        self.assertFalse(translator.calls[0][1]["full_page"])

    def test_parallel_queue_commits_each_page_before_translating_next(self):
        first_blocks = [TextBlock(text=["one"])]
        second_blocks = [TextBlock(text=["two"])]
        project = FakeProject(
            {"001.png": first_blocks, "002.png": second_blocks}
        )
        translator = FakeTranslator()
        thread = TranslateThread()
        thread.translator = translator
        thread.module = translator
        thread.imgtrans_proj = project
        thread.num_process_pages = 2
        thread.pipeline_pagekey_queue = ["001.png", "002.png"]

        thread._run_translate_pipeline()

        self.assertEqual(
            translator.event_snapshots,
            [
                [("begin", "001.png")],
                [
                    ("begin", "001.png"),
                    ("finish", "001.png", "한국어"),
                    ("begin", "002.png"),
                ],
            ],
        )
        self.assertEqual(
            project.events,
            [
                ("begin", "001.png"),
                ("finish", "001.png", "한국어"),
                ("begin", "002.png"),
                ("finish", "002.png", "한국어"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
