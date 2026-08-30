import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ui.module_manager import (
    ImgtransThread,
    InpaintThread,
    ModuleManager,
    OCRThread,
    TextDetectThread,
    TranslateThread,
    cfg_module,
    translate_project_textblocks,
)
from ui.mainwindow import MainWindow
from utils.textblock import TextBlock


class FakeProject:
    def __init__(self, pages):
        self.pages = pages
        self.events = []
        self.progress = []

    def begin_full_page_translation(self, page_key):
        self.events.append(("begin", page_key))

    def mark_translation_finished(self, page_key, target):
        self.events.append(("finish", page_key, target))

    def read_img(self, _page_key):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def update_page_progress(self, page_key, progress):
        self.progress.append((page_key, progress))

    def load_mask_by_imgname(self, _page_key):
        return None


class FakeTranslator:
    lang_target = "한국어"

    def __init__(
        self,
        success=True,
        error=None,
        *,
        low_vram_mode=False,
        computationally_intensive=False,
    ):
        self.success = success
        self.error = error
        self.low_vram_mode = low_vram_mode
        self.computationally_intensive = computationally_intensive
        self.calls = []
        self.event_snapshots = []

    def delay(self):
        return 0

    def is_computational_intensive(self):
        return self.computationally_intensive

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
    def make_imgtrans_thread(self, project, translator):
        textdetect_thread = TextDetectThread()
        ocr_thread = OCRThread()
        ocr_thread.module = SimpleNamespace(run_ocr=MagicMock())
        ocr_thread.ocr = ocr_thread.module
        translate_thread = TranslateThread()
        translate_thread.translator = translator
        translate_thread.module = translator
        inpaint_thread = InpaintThread()
        pipeline = ImgtransThread(
            textdetect_thread,
            ocr_thread,
            translate_thread,
            inpaint_thread,
        )
        pipeline.imgtrans_proj = project
        pipeline.pages_to_process = None
        pipeline.process_idx_to_page_idx = {}
        return pipeline

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

    def test_standalone_wrapper_emits_once_without_error_dialog_on_success(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        thread = TranslateThread()
        emitted = []
        thread.finish_translate_page.connect(emitted.append)

        with patch.object(
            thread,
            "_translate_page",
            return_value=True,
        ) as translate_page, patch.object(
            thread,
            "start",
            side_effect=lambda: thread.job(),
        ), patch(
            "ui.module_manager.create_error_dialog",
        ) as error_dialog:
            thread.translatePage(project, "001.png")

        translate_page.assert_called_once_with(
            project,
            "001.png",
            emit_finished=False,
        )
        self.assertEqual(emitted, ["001.png"])
        error_dialog.assert_not_called()

    def test_standalone_wrapper_reports_exception_and_still_emits_once(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        thread = TranslateThread()
        emitted = []
        thread.finish_translate_page.connect(emitted.append)
        error = RuntimeError("translation failed")

        with patch.object(
            thread,
            "_translate_page",
            side_effect=error,
        ), patch.object(
            thread,
            "start",
            side_effect=lambda: thread.job(),
        ), patch(
            "ui.module_manager.create_error_dialog",
        ) as error_dialog:
            thread.translatePage(project, "001.png")

        self.assertEqual(emitted, ["001.png"])
        error_dialog.assert_called_once()
        self.assertIs(error_dialog.call_args.args[0], error)
        self.assertEqual(error_dialog.call_args.args[2], "TranslationFailed")

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

    def test_mainwindow_and_module_manager_forward_selected_page_context(self):
        block = TextBlock(text=["one"])
        project = FakeProject({"001.png": [block]})
        project.current_img = "001.png"
        project.img_array = np.zeros((4, 4, 3), dtype=np.uint8)
        project.mask_array = None
        translator = FakeTranslator()
        pipeline = self.make_imgtrans_thread(project, translator)

        class ManagerHarness:
            runBlktransPipeline = ModuleManager.runBlktransPipeline
            _startBlktransPipeline = ModuleManager._startBlktransPipeline

            def __init__(self):
                self.imgtrans_proj = project
                self.imgtrans_thread = pipeline
                self.prepare_msgbox = None
                self.progress_msgbox = SimpleNamespace(
                    hide=MagicMock(),
                    hide_all_bars=MagicMock(),
                    zero_progress=MagicMock(),
                    show=MagicMock(),
                    ocr_bar=MagicMock(),
                    translate_bar=MagicMock(),
                    inpaint_bar=MagicMock(),
                )

            @staticmethod
            def terminateRunningThread():
                return True

            @staticmethod
            def _prepare_modules_then(_required_modules, callback):
                callback()

        manager = ManagerHarness()
        source_editor = SimpleNamespace(toPlainText=lambda: "Current")
        block_item = SimpleNamespace(
            blk=block,
            idx=0,
            absBoundingRect=lambda: [0, 0, 2, 2],
        )
        window = SimpleNamespace(
            imgtrans_proj=project,
            module_manager=manager,
            global_search_widget=SimpleNamespace(
                set_document_edited=MagicMock()
            ),
            st_manager=SimpleNamespace(
                pairwidget_list=[SimpleNamespace(e_source=source_editor)]
            ),
        )

        with patch.object(
            pipeline,
            "start",
            side_effect=lambda: pipeline.job(),
        ):
            started = MainWindow.translateBlkitemList(
                window,
                [block_item],
                1,
            )

        self.assertTrue(started)
        self.assertIs(pipeline.imgtrans_proj, project)
        self.assertIs(translator.calls[0][1]["project"], project)
        self.assertEqual(translator.calls[0][1]["page_key"], "001.png")

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

    def test_direct_pipeline_forwards_context_in_strict_and_serial_modes(self):
        cases = (
            ("strict", True, False),
            ("serial", False, True),
        )
        for label, enable_inpaint, computationally_intensive in cases:
            with self.subTest(label=label):
                blocks = [TextBlock(text=["one"])]
                project = FakeProject({"001.png": blocks})
                translator = FakeTranslator(
                    computationally_intensive=computationally_intensive,
                )
                pipeline = self.make_imgtrans_thread(project, translator)

                with patch.multiple(
                    cfg_module,
                    enable_detect=False,
                    enable_ocr=False,
                    enable_translate=True,
                    enable_inpaint=enable_inpaint,
                ):
                    pipeline._imgtrans_pipeline()

                self.assertEqual(len(translator.calls), 1)
                self.assertIs(translator.calls[0][1]["project"], project)
                self.assertEqual(
                    translator.calls[0][1]["page_key"],
                    "001.png",
                )
                self.assertTrue(translator.calls[0][1]["full_page"])

    def test_low_vram_deferred_pipeline_forwards_project_and_page(self):
        blocks = [TextBlock(text=["one"])]
        project = FakeProject({"001.png": blocks})
        translator = FakeTranslator(low_vram_mode=True)
        pipeline = self.make_imgtrans_thread(project, translator)

        with patch.multiple(
            cfg_module,
            enable_detect=False,
            enable_ocr=False,
            enable_translate=True,
            enable_inpaint=False,
        ), patch(
            "ui.module_manager.unload_modules",
        ) as unload:
            pipeline._imgtrans_pipeline()

        self.assertEqual(len(translator.calls), 1)
        self.assertIs(translator.calls[0][1]["project"], project)
        self.assertEqual(translator.calls[0][1]["page_key"], "001.png")
        self.assertTrue(translator.calls[0][1]["full_page"])
        unload.assert_called_once_with(
            pipeline,
            ["textdetector", "inpainter", "ocr"],
        )


if __name__ == "__main__":
    unittest.main()
