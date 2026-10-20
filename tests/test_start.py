"""启动器的安装失败恢复与跨目录启动，不实际联网安装。"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import start


class LauncherTests(unittest.TestCase):
    def test_environment_creation_and_requirement_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            requirements = project / "requirements.txt"
            requirements.write_text("requests\n", encoding="utf-8")

            def create_environment(path):
                python = start.environment_python(project)
                python.parent.mkdir(parents=True)
                python.touch()

            with patch("start.venv.EnvBuilder.create", side_effect=create_environment) as create, \
                 patch("start.subprocess.run") as run:
                python = start.prepare_environment(project)
                start.prepare_environment(project)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][:4], [str(python), "-m", "pip", "install"])
                requirements.write_text("requests>=2.31\n", encoding="utf-8")
                start.prepare_environment(project)
                self.assertEqual(run.call_count, 2)

    def test_failed_install_is_retried_without_a_success_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "requirements.txt").write_text("requests\n", encoding="utf-8")
            python = start.environment_python(project)
            python.parent.mkdir(parents=True)
            python.touch()
            with patch("start.subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
                with self.assertRaises(subprocess.CalledProcessError):
                    start.prepare_environment(project)
            self.assertFalse((project / ".venv" / ".astockai-requirements").exists())
            with patch("start.subprocess.run") as run:
                start.prepare_environment(project)
                run.assert_called_once()

    def test_launch_uses_project_directory_and_localhost(self):
        with patch("start.prepare_environment", return_value=Path("/fake/python")), \
             patch("start.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(start.main(["--port", "8502"]), 0)
            self.assertEqual(run.call_args.kwargs["cwd"], start.PROJECT)
            self.assertIn("--server.address=127.0.0.1", run.call_args.args[0])
            self.assertIn("--server.port=8502", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
