"""本机后台任务状态与编排测试，不启动真实进程或网络请求。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from background_tasks import (
    get_active_task,
    read_task_log_tail,
    run_background_task,
    start_background_task,
    task_log_file,
    task_status_file,
)


class FakeProcess:
    pid = 43210


class BackgroundTaskTests(unittest.TestCase):
    def test_start_persists_queued_task_and_launches_isolated_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def process_factory(*args, **kwargs):
                calls.append((args, kwargs))
                return FakeProcess()

            result = start_background_task("research_refresh", root, process_factory=process_factory)
            task = result["task"]
            status_file = task_status_file(root, task["任务编号"])

            self.assertEqual(result["status"], "started")
            self.assertTrue(status_file.is_file())
            self.assertEqual(task["进程ID"], 43210)
            self.assertEqual(calls[0][0][0][1:3], ["background_task_runner.py", "research_refresh"])
            self.assertTrue(calls[0][1]["start_new_session"])
            with patch("background_tasks._pid_is_running", return_value=True):
                self.assertEqual(get_active_task(root)["任务编号"], task["任务编号"])

    def test_data_update_records_each_step_without_running_research(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_id = "data-task"
            with patch("background_tasks.run_update_data", return_value={"status": "partial", "message": "部分股票失败"}), \
                 patch("background_tasks.download_benchmarks", return_value=[
                     {"名称": "沪深300", "status": "success"},
                     {"名称": "中证1000", "status": "success"},
                 ]), \
                 patch("background_tasks.refresh_research_artifacts") as refresh:
                result = run_background_task("data_update", task_id, root)

            self.assertEqual(result["状态"], "partial")
            self.assertEqual([step["步骤"] for step in result["步骤"]], ["股票日线更新", "市场基准更新"])
            refresh.assert_not_called()
            saved = task_status_file(root, task_id).read_text(encoding="utf-8")
            self.assertIn("部分股票失败", saved)
            self.assertIn("不训练模型", result["边界"])

    def test_research_task_and_log_tail_are_visible_after_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_id = "research-task"
            with patch("background_tasks.refresh_research_artifacts", return_value={
                "步骤": [
                    {"步骤": "数据审计", "状态": "success", "说明": "完成", "耗时秒": 0.1},
                    {"步骤": "训练", "状态": "insufficient", "说明": "样本不足", "耗时秒": 0.2},
                ],
                "边界": "只使用本地数据。",
            }):
                result = run_background_task("research_refresh", task_id, root)
            log_file = task_log_file(root, task_id)
            log_file.write_text("前文\n最后一行", encoding="utf-8")
            result["日志文件"] = str(log_file)

            self.assertEqual(result["状态"], "partial")
            self.assertEqual(result["步骤"][1]["状态"], "insufficient")
            self.assertEqual(read_task_log_tail(root, result), "前文\n最后一行")


if __name__ == "__main__":
    unittest.main()
