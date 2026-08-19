import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from announcement_evidence import collect_announcement_snapshot
from analysis_evidence import load_announcement_context


class FakeClient:
    def get_announcements(self, code, count):
        return pd.DataFrame([{
            "date": "2026-08-18",
            "title": "测试公司关于回购股份的进展公告",
            "url": f"https://example.test/{code}",
        }])


class AnnouncementEvidenceTests(unittest.TestCase):
    def test_snapshot_keeps_official_facts_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watchlist = root / "watchlist.json"
            output = root / "output"
            watchlist.write_text(json.dumps({"stocks": [{"code": "000001", "name": "测试公司"}]}), encoding="utf-8")
            snapshot_file = collect_announcement_snapshot(watchlist, output, "2026-08-18", FakeClient())
            snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
            item = snapshot["stocks"][0]["公告"][0]
            self.assertEqual(item["类别"], "权益与融资")
            self.assertEqual(item["官方链接"], "https://example.test/000001")
            context = load_announcement_context(output, "2026-08-18")
            self.assertTrue(context["available"])

    def test_missing_same_day_snapshot_is_not_replaced_with_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            context = load_announcement_context(Path(directory), "2026-08-18")
            self.assertFalse(context["available"])
            self.assertIn("当日", context["status"])


if __name__ == "__main__":
    unittest.main()
