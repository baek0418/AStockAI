"""研究产物显式刷新编排测试。"""

import unittest
from unittest.mock import patch

from research_refresh import refresh_research_artifacts


class ResearchRefreshTests(unittest.TestCase):
    def test_refresh_runs_all_local_research_steps_in_order(self):
        calls = []

        def audit(project):
            calls.append(("audit", project))
            return {}, "audit.json", "audit.md"

        def train(project):
            calls.append(("train", project))
            return {"status": "success", "message": "训练完成"}

        def compare(project):
            calls.append(("compare", project))
            return {}, "comparison.json", "comparison.md"

        def portfolio(project):
            calls.append(("portfolio", project))
            return {"status": "success", "message": "回测完成"}

        with patch("research_refresh.run_audit", audit), \
             patch("research_refresh.train_prediction_v2", train), \
             patch("research_refresh.create_comparison", compare), \
             patch("research_refresh.run_oos_portfolio_research", portfolio):
            result = refresh_research_artifacts("/tmp/research-project")

        self.assertEqual([name for name, _ in calls], ["audit", "train", "compare", "portfolio"])
        self.assertEqual(result["状态"], "success")
        self.assertEqual([step["状态"] for step in result["步骤"]], ["success"] * 4)
        self.assertIn("不下载行情", result["边界"])

    def test_insufficient_or_failed_step_is_reported_without_skipping_later_steps(self):
        with patch("research_refresh.run_audit", return_value=({}, "audit.json", "audit.md")), \
             patch("research_refresh.train_prediction_v2", return_value={"status": "insufficient", "message": "样本不足"}), \
             patch("research_refresh.create_comparison", return_value=({}, "comparison.json", "comparison.md")), \
             patch("research_refresh.run_oos_portfolio_research", side_effect=ValueError("回测数据不足")):
            result = refresh_research_artifacts("/tmp/research-project")

        self.assertEqual(result["状态"], "failed")
        self.assertEqual(result["步骤"][1]["状态"], "insufficient")
        self.assertEqual(result["步骤"][3]["状态"], "failed")
        self.assertIn("回测数据不足", result["步骤"][3]["说明"])


if __name__ == "__main__":
    unittest.main()
