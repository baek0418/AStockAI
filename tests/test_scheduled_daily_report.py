"""交易日日报定时任务测试，不会调用真实流水线或 SMTP。"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from run_scheduled_daily_report import (
    get_trading_day_status,
    load_market_holidays,
    run_scheduled_report,
)


BEIJING = ZoneInfo("Asia/Shanghai")


class ScheduledDailyReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_directory = Path(self.temporary_directory.name)
        (self.project_directory / ".env").write_text("SMTP_PASSWORD=not-used", encoding="utf-8")
        python_executable = self.project_directory / ".venv" / "bin" / "python"
        python_executable.parent.mkdir(parents=True)
        python_executable.write_text("", encoding="utf-8")
        self.calendar_file = self.project_directory / "calendar.json"
        self.calendar_file.write_text(
            json.dumps({"holidays": {"2026": ["2026-10-01"]}}),
            encoding="utf-8",
        )
        self.logger = Mock()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_weekend_holiday_and_missing_year_are_not_trading_days(self):
        calendar = load_market_holidays(self.calendar_file)

        self.assertFalse(get_trading_day_status(datetime(2026, 10, 3).date(), calendar)[0])
        self.assertFalse(get_trading_day_status(datetime(2026, 10, 1).date(), calendar)[0])
        self.assertFalse(get_trading_day_status(datetime(2027, 1, 4).date(), calendar)[0])
        self.assertTrue(get_trading_day_status(datetime(2026, 10, 8).date(), calendar)[0])

    def test_non_trading_day_does_not_run_pipeline_or_send_email(self):
        runner = Mock()
        result = run_scheduled_report(
            now=datetime(2026, 10, 1, 9, 30, tzinfo=BEIJING),
            runner=runner,
            project_directory=self.project_directory,
            calendar_file=self.calendar_file,
            logger=self.logger,
        )

        self.assertEqual(result, 0)
        runner.assert_not_called()

    def test_trading_day_runs_existing_pipeline_with_send_email_flag(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0))
        result = run_scheduled_report(
            now=datetime(2026, 10, 8, 9, 30, tzinfo=BEIJING),
            runner=runner,
            project_directory=self.project_directory,
            calendar_file=self.calendar_file,
            logger=self.logger,
        )

        self.assertEqual(result, 0)
        runner.assert_called_once_with(
            [
                str(self.project_directory / ".venv" / "bin" / "python"),
                "pipeline.py",
                "--send-email",
            ],
            cwd=self.project_directory,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
