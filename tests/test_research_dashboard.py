"""研究总览只读汇总测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from astock_core.research.research_dashboard import (
    build_research_dashboard,
    build_research_workbench_summary,
    build_user_system_status,
    research_dashboard_source_mtime,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class ResearchDashboardTests(unittest.TestCase):
    def test_missing_reports_are_explicit_and_never_mark_research_as_displayable(self):
        with tempfile.TemporaryDirectory() as directory:
            dashboard = build_research_dashboard(Path(directory))

        self.assertEqual(dashboard["数据健康"]["状态"], "数据不足")
        self.assertEqual(dashboard["模型验证"]["状态"], "数据不足")
        self.assertEqual(dashboard["组合回测"]["状态"], "数据不足")
        self.assertFalse(dashboard["总览结论"]["研究结果可展示"])

    def test_dashboard_aggregates_local_reports_without_approving_probability_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "output/prediction/prediction_data_audit_2026-08-20.json", {
                "生成时间": "2026-08-20 10:00:00",
                "股票数量": 2,
                "横截面限制": "覆盖有限。",
                "特征构建跳过文件": [{"file": "坏文件.csv", "reason": "缺少字段"}],
                "文件审计": [
                    {"文件": "甲历史.csv", "状态": "可训练", "日期范围": ["2025-01-01", "2026-08-20"]},
                    {"文件": "乙历史.csv", "状态": "无效", "日期范围": [None, None]},
                    {"文件": "池外.csv", "状态": "研究池外", "日期范围": ["2025-01-01", "2026-08-19"]},
                ],
            })
            write_json(root / "output/prediction/prediction_outperform_evaluation_2026-08-20.json", {
                "模型版本": "v5.1-test",
                "生成时间": "2026-08-20 10:00:00",
                "验证状态": "滚动验证完成。",
                "数据范围": {"训练日期范围": ["2024-01-01", "2026-08-20"], "股票数量": 2, "样本数": 200},
                "汇总样本外指标": {"brier_score": 0.20, "log_loss": 0.5, "roc_auc": 0.60, "accuracy": 0.55},
                "汇总朴素概率基线": {"brier_score": 0.21, "log_loss": 0.6, "roc_auc": 0.5, "accuracy": 0.5},
                "滚动样本外验证": [
                    {"status": "success", "校准": {"calibrated": True}},
                    {"status": "success", "校准": {"calibrated": False}},
                ],
                "风险提示": ["仅作研究。"],
            })
            write_json(root / "models/predict_5d_outperform_benchmark_metadata.json", {"滚动验证通过": True})
            write_json(root / "output/prediction/prediction_comparison_2026-08-20.json", {
                "结论": {"可接入日报或 Web": False, "结论": "保持研究隔离。"},
            })
            write_json(root / "output/portfolio/oos_portfolio_research_2026-08-20.json", {
                "策略": "v5.1 样本外组合",
                "市场基准": "沪深300",
                "组合统计": {"累计收益率": 0.1, "超额累计收益率": -0.02, "最大回撤": -0.05, "交易笔数": 12},
                "参数": {"max_positions": 5},
                "信号覆盖诊断": {"执行信号数": 40},
            })

            dashboard = build_research_dashboard(root)
            source_mtime = research_dashboard_source_mtime(root)

        health = dashboard["数据健康"]
        model = dashboard["模型验证"]
        portfolio = dashboard["组合回测"]
        self.assertEqual(health["可训练股票数"], 2)
        self.assertEqual(health["最新日线日期"], "2026-08-20")
        self.assertEqual(health["数据问题文件数"], 1)
        self.assertEqual(health["研究池外文件数"], 1)
        self.assertEqual(model["状态"], "可用")
        self.assertTrue(model["技术验证通过"])
        self.assertFalse(model["允许展示概率"])
        self.assertEqual(model["完成窗口数"], 2)
        self.assertEqual(model["完成校准窗口数"], 1)
        self.assertEqual(portfolio["统计"]["交易笔数"], 12)
        self.assertIsInstance(source_mtime, int)

    def test_workbench_keeps_an_unapproved_model_out_of_investment_display(self):
        summary = build_research_workbench_summary(
            {
                "数据健康": {"状态": "可用", "时效滞后日": 0, "时效说明": "同日。"},
                "模型验证": {"状态": "可用", "技术验证通过": True, "允许展示概率": False, "展示边界": "保持研究隔离。"},
                "组合回测": {"状态": "可用"},
            }
        )

        self.assertEqual(summary["研究决策"], "模型仍保持研究隔离")
        self.assertIn("人工复核", summary["下一步"])
        self.assertEqual(summary["提示级别"], "warning")

    def test_user_status_explains_that_an_unapproved_model_does_not_block_daily_use(self):
        status = build_user_system_status(
            {
                "数据健康": {"状态": "可用", "时效滞后日": 0},
                "模型验证": {"技术验证通过": False, "允许展示概率": False},
            }
        )

        self.assertEqual(status["数据状态"], "可用")
        self.assertEqual(status["模型状态"], "暂不显示模型预测")
        self.assertIn("不影响", status["模型说明"])


if __name__ == "__main__":
    unittest.main()
