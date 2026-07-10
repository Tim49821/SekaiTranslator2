import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from shutil import rmtree

from utils.updater import BallonsTranslatorUpdater, ReleaseInfo


class UpdaterTest(unittest.TestCase):
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
