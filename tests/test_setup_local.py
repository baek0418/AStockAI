import json
import tempfile
import unittest
from pathlib import Path

from setup_local import run_first_setup


class SetupLocalTests(unittest.TestCase):
    def test_first_setup_creates_only_an_empty_local_watchlist_and_disables_ai_email(self):
        calls = []

        def fake_pipeline(**kwargs):
            calls.append(kwargs)
            return {"success": True, "steps": []}

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "watchlist.example.json").write_text('{"stocks": []}', encoding="utf-8")
            result = run_first_setup(project, pipeline_runner=fake_pipeline)
            saved = json.loads((project / "watchlist.json").read_text(encoding="utf-8"))

        self.assertTrue(result["success"])
        self.assertTrue(result["watchlist_created"])
        self.assertEqual(saved, {"stocks": []})
        self.assertEqual(calls, [{"no_ai": True, "send_email": False}])

    def test_existing_local_watchlist_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "watchlist.example.json").write_text('{"stocks": []}', encoding="utf-8")
            (project / "watchlist.json").write_text('{"stocks": [{"code": "600000"}]}', encoding="utf-8")
            result = run_first_setup(project, pipeline_runner=lambda **kwargs: {"success": True})
            saved = json.loads((project / "watchlist.json").read_text(encoding="utf-8"))

        self.assertFalse(result["watchlist_created"])
        self.assertEqual(saved, {"stocks": [{"code": "600000"}]})


if __name__ == "__main__":
    unittest.main()
