import unittest
from collections import OrderedDict
from unittest.mock import patch

from modules.translators.base import (
    BaseTranslator,
    translation_is_successful,
    translation_request_covers_full_page,
)
from utils.config import RunStatus
from utils.proj_imgtrans import ProjImgTrans
from utils.textblock import TextBlock


class ProjectTranslationContextTest(unittest.TestCase):
    def test_identity_changes_when_project_contents_are_reloaded(self):
        project = ProjImgTrans()
        project.directory = "/tmp/demo-project"
        first = project.load_identity

        with patch("utils.proj_imgtrans.find_all_imgs", return_value=[]):
            project.load_from_dict({"pages": {}, "image_info": {}})

        self.assertIsNot(project.load_identity, first)

    def test_identity_changes_when_new_project_is_created(self):
        project = ProjImgTrans()
        project.directory = "/tmp/demo-project"
        first = project.load_identity

        with patch("utils.proj_imgtrans.osp.exists", return_value=True), \
             patch("utils.proj_imgtrans.find_all_imgs", return_value=[]), \
             patch.object(project, "save"):
            project.new_project()

        self.assertIsNot(project.load_identity, first)

    def test_identity_is_not_serialized(self):
        project = ProjImgTrans()

        self.assertNotIn("load_identity", project.to_dict())
        self.assertNotIn("_load_identity", project.to_dict())

    def test_begin_and_finish_manage_target_metadata(self):
        project = ProjImgTrans()
        project.pages = OrderedDict((
            ("001.png", [TextBlock(text=["hello"])]),
        ))
        project._image_info = {
            "001.png": {
                "finish_code": RunStatus.FIN_TRANSLATE,
                "translation_target": "English",
            }
        }

        project.begin_full_page_translation("001.png")
        self.assertFalse(
            project._image_info["001.png"]["finish_code"]
            & RunStatus.FIN_TRANSLATE
        )
        self.assertNotIn("translation_target", project._image_info["001.png"])

        project.mark_translation_finished("001.png", "한국어")
        self.assertTrue(
            project._image_info["001.png"]["finish_code"]
            & RunStatus.FIN_TRANSLATE
        )
        self.assertEqual(
            project._image_info["001.png"]["translation_target"],
            "한국어",
        )

    def test_set_page_progress_removes_target_when_translation_is_cleared(self):
        project = ProjImgTrans()
        project._image_info = {
            "001.png": {
                "finish_code": RunStatus.FIN_ALL,
                "translation_target": "English",
            }
        }

        project.set_page_progress(
            "001.png",
            RunStatus.FIN_ALL & ~RunStatus.FIN_TRANSLATE,
        )

        self.assertNotIn("translation_target", project._image_info["001.png"])


class RecordingTranslator(BaseTranslator):
    concate_text = False
    params = {}

    def _setup_translator(self):
        self.lang_map["日本語"] = "Japanese"
        self.lang_map["한국어"] = "Korean"
        self.context_call = None

    def _translate(self, src_list):
        return ["번역:" + text for text in src_list]

    def _translate_with_context(
        self,
        src_list,
        *,
        project=None,
        page_key=None,
        commit_history_window=False,
    ):
        self.context_call = (project, page_key, commit_history_window)
        return self._translate(src_list)


class TranslatorContextBoundaryTest(unittest.TestCase):
    def test_textblock_boundary_forwards_project_page_and_completion(self):
        translator = RecordingTranslator("日本語", "한국어")
        blocks = [TextBlock(text=["one"]), TextBlock(text=[""])]
        project = type("Project", (), {"pages": {"001.png": blocks}})()

        success = translator.translate_textblk_lst(
            blocks,
            project=project,
            page_key="001.png",
            full_page=True,
        )

        self.assertTrue(success)
        self.assertEqual(translator.context_call, (project, "001.png", True))
        self.assertEqual(blocks[0].translation, "번역:one")
        self.assertEqual(blocks[1].translation, "")

    def test_full_page_detection_uses_block_identity_for_source_blocks(self):
        selected = TextBlock(text=["same"])
        same_text_different_block = TextBlock(text=["same"])
        empty_unselected = TextBlock(text=[""])
        project = type(
            "Project",
            (),
            {"pages": {"001.png": [selected, empty_unselected]}},
        )()

        self.assertTrue(
            translation_request_covers_full_page(
                [selected], project, "001.png"
            )
        )
        self.assertFalse(
            translation_request_covers_full_page(
                [same_text_different_block], project, "001.png"
            )
        )
        self.assertTrue(
            translation_request_covers_full_page(
                [], project, "missing.png", full_page=True
            )
        )

    def test_success_uses_sources_captured_before_preprocess_hooks(self):
        translator = RecordingTranslator("日本語", "한국어")
        blocks = [TextBlock(text=["source"])]

        def clear_source_text(translations, source_text, **kwargs):
            source_text[0] = ""
            translations[0] = ""

        with patch.object(
            translator,
            "_preprocess_hooks",
            OrderedDict((("clear_source", clear_source_text),)),
        ):
            success = translator.translate_textblk_lst(blocks)

        self.assertFalse(success)

    def test_error_markers_and_empty_outputs_are_not_successful(self):
        self.assertTrue(translation_is_successful("source", "target"))
        self.assertFalse(translation_is_successful("source", ""))
        self.assertFalse(
            translation_is_successful("source", "[ERROR: API Failed]")
        )
        self.assertTrue(translation_is_successful("", ""))


if __name__ == "__main__":
    unittest.main()
