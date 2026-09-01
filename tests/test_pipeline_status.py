"""流水线日报状态测试：不运行网络、研究或邮件任务。"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pipeline import get_daily_report_status


class DailyReportPipelineStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.report_file = Path(self.temporary_directory.name) / "日报.md"
        self.report_file.write_text("# 日报\n\n规则分析。\n", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def step_record(self, audit):
        return {
            "start_time": datetime.now(),
            "result": {
                "output_file": self.report_file,
                "quote_provenance": audit,
            },
        }

    def test_complete_quote_audit_is_saved_without_downgrading_report(self):
        result = get_daily_report_status(
            self.step_record(
                {
                    "日报快照日期": "2026-09-01",
                    "关注股数": 2,
                    "可核对数": 2,
                    "同日记录数": 2,
                    "较新记录数": 0,
                    "落后记录数": 0,
                    "未记录数": 0,
                    "备用源更新数": 1,
                    "重试后成功数": 1,
                    "状态": "可核对：关注股行情来源记录与日报快照日期一致。",
                    "股票": [{"不应": "写入流水线摘要"}],
                }
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["details"]["quote_provenance"]["可核对数"], 2)
        self.assertNotIn("股票", result["details"]["quote_provenance"])

    def test_incomplete_quote_audit_marks_report_partial_without_blocking_output(self):
        result = get_daily_report_status(
            self.step_record(
                {
                    "日报快照日期": "2026-09-01",
                    "关注股数": 2,
                    "可核对数": 1,
                    "同日记录数": 1,
                    "较新记录数": 0,
                    "落后记录数": 0,
                    "未记录数": 1,
                    "备用源更新数": 0,
                    "重试后成功数": 0,
                    "状态": "审计不完整：不因缺失或落后的来源记录扩大日报结论。",
                }
            )
        )

        self.assertEqual(result["status"], "partial")
        self.assertIn("行情来源审计不完整", result["error"])
        self.assertEqual(result["output_file"], str(self.report_file))


if __name__ == "__main__":
    unittest.main()
