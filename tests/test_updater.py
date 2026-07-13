import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from unittest.mock import patch

import launch
from utils.updater import BallonsTranslatorUpdater, ReleaseInfo


class UpdaterTest(unittest.TestCase):
    def test_source_update_defaults_to_fork_main_branch(self):
        self.assertEqual(launch.UPDATE_REMOTE, "origin")
        self.assertEqual(launch.BRANCH, "main")

    def test_git_ahead_behind_counts_each_direction(self):
        with patch.object(launch, "run", side_effect=["2\n", "3\n"]) as run_mock:
            result = launch.git_ahead_behind("HEAD", "origin/main")

        self.assertEqual(result, (2, 3))
        self.assertEqual(
            [call.args[0] for call in run_mock.call_args_list],
            [
                [launch.git, "rev-list", "--count", "origin/main..HEAD"],
                [launch.git, "rev-list", "--count", "HEAD..origin/main"],
            ],
        )

    def test_source_update_only_fast_forwards_when_local_has_no_unique_commits(self):
        self.assertEqual(launch.source_update_action(ahead=0, behind=2), "fast_forward")
        self.assertEqual(launch.source_update_action(ahead=1, behind=2), "diverged")
        self.assertEqual(launch.source_update_action(ahead=1, behind=0), "ahead")
        self.assertEqual(launch.source_update_action(ahead=0, behind=0), "up_to_date")

    def test_headless_package_install_requires_explicit_opt_in(self):
        config = SimpleNamespace(
            package_manager=SimpleNamespace(auto_install_missing_packages=True),
        )

        self.assertFalse(launch.apply_package_install_policy(config, headless=True, allow_package_install=False))
        self.assertTrue(launch.apply_package_install_policy(config, headless=True, allow_package_install=True))

    def test_gui_package_install_policy_preserves_saved_choice(self):
        config = SimpleNamespace(
            package_manager=SimpleNamespace(auto_install_missing_packages=True),
        )

        self.assertTrue(launch.apply_package_install_policy(config, headless=False, allow_package_install=False))

    def test_apply_update_is_guarded_for_flat_layout_fork(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            updater = BallonsTranslatorUpdater(program_path=temp_dir, cache_dir=temp_dir)
            release = ReleaseInfo(
                tag_name="v1.5.6",
                version="1.5.6",
                html_url="https://example.invalid/release",
                zip_url="https://example.invalid/source.zip",
            )

            result = updater.apply_update(release, current_version="1.4.0")

        self.assertEqual(result.status, "manual_update_required")
        self.assertIn("SEKAI_TRANSLATOR_ALLOW_UPSTREAM_SELF_UPDATE", result.git_message)

    def test_version_falls_back_to_launch_py(self):
        temp_dir = tempfile.mkdtemp(dir=".")
        try:
            temp_dir_path = Path(temp_dir).resolve()
            launch_path = temp_dir_path / "launch.py"
            with launch_path.open("w", encoding="utf8") as f:
                f.write("VERSION = '9.8.7'\n")

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from utils.version import get_current_version; print(get_current_version(sys.argv[1]))",
                    str(temp_dir_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).resolve().parents[1]),
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "9.8.7")
        finally:
            rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
