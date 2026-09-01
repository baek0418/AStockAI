"""未来收益区间实验的无泄漏窗口、覆盖率和准入边界测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from return_interval_research import (
    TARGET_COVERAGE,
    _cqr_scores,
    _interval_metrics,
    _volatility_scale,
    assess_admission,
    audit_price_basis,
    evaluate_interval_windows,
    select_interval_method,
)
from prediction_features import FEATURE_COLUMNS, RETURN_COLUMN


class ReturnIntervalResearchTests(unittest.TestCase):
    def test_interval_metrics_measures_coverage_and_width(self):
        metrics = _interval_metrics([0.01, -0.02, 0.03], [-0.01, -0.03, 0.0], [0.02, -0.01, 0.02])
        self.assertEqual(metrics["样本数"], 3)
        self.assertAlmostEqual(metrics["覆盖率"], 2 / 3, places=6)
        self.assertAlmostEqual(metrics["平均区间宽度"], 0.023333, places=6)

    def test_admission_rejects_when_price_history_is_not_point_in_time_auditable(self):
        window = {
            "状态": "success",
            "区间模型": {"样本数": 100, "覆盖率": TARGET_COVERAGE, "平均区间宽度": 0.1, "中心预测MAE": 0.03},
            "无条件收益率基线": {"样本数": 100, "覆盖率": 0.75, "平均区间宽度": 0.12, "中心预测MAE": 0.04},
            "历史波动率基线": {"样本数": 100, "覆盖率": 0.78, "平均区间宽度": 0.11, "中心预测MAE": 0.04},
        }
        admission = assess_admission({"严格时点价格依据可用": False}, [window, window, window])
        self.assertFalse(admission["是否准入"])
        self.assertIn("时点可得性", admission["拒绝原因"][0])

    def test_windows_keep_a_five_day_gap_between_training_and_test_dates(self):
        dates = pd.bdate_range("2024-01-01", periods=120)
        frame = pd.DataFrame(
            [
                {
                    "日期": date,
                    "股票代码": f"{stock:06d}",
                    "股票名称": f"测试{stock}",
                    RETURN_COLUMN: 0.01 * np.sin(index / 5 + stock),
                    **{column: 0.1 + index / 1000 for column in FEATURE_COLUMNS},
                }
                for stock in range(1, 9)
                for index, date in enumerate(dates)
            ]
        )
        windows = evaluate_interval_windows(frame)
        successful = [item for item in windows if item["状态"] == "success"]
        self.assertGreaterEqual(len(successful), 1)
        for item in successful:
            train_end = pd.Timestamp(item["训练日期范围"][1])
            test_start = pd.Timestamp(item["测试日期范围"][0])
            self.assertGreaterEqual((test_start - train_end).days, 7)
            self.assertEqual(item["内部隔离交易日数"], 5)

    def test_volatility_scale_widens_the_interval_for_more_volatile_rows(self):
        frame = pd.DataFrame({"volatility_20d": [0.01, 0.04]})
        scales = _volatility_scale(frame)
        self.assertGreater(scales[1], scales[0])
        self.assertAlmostEqual(scales[0], 0.01 * np.sqrt(5))

    def test_price_audit_requires_explicit_raw_snapshot_provenance(self):
        dataset = pd.DataFrame({"股票代码": ["600000"], "股票名称": ["测试股"]})
        with tempfile.TemporaryDirectory() as directory:
            provenance_directory = Path(directory) / "provenance"
            provenance_directory.mkdir()
            (provenance_directory / "600000.json").write_text(
                json.dumps({"数据源": "BaoStock raw kline", "复权方式": "raw"}), encoding="utf-8"
            )
            audit = audit_price_basis(directory, dataset)
        self.assertTrue(audit["严格时点价格依据可用"])
        self.assertEqual(audit["非原始价格记录数"], 0)

    def test_cqr_scores_are_zero_or_negative_inside_and_positive_outside_the_bounds(self):
        scores = _cqr_scores([0.0, -0.03, 0.04], [-0.01, -0.02, -0.01], [0.01, 0.02, 0.03])
        self.assertLessEqual(scores[0], 0)
        self.assertAlmostEqual(scores[1], 0.01)
        self.assertAlmostEqual(scores[2], 0.01)

    def test_selection_falls_back_to_simple_volatility_range_when_cqr_does_not_win(self):
        window = {
            "状态": "success",
            "区间模型": {"样本数": 10, "覆盖率": 0.79, "平均区间宽度": 0.12, "中心预测MAE": 0.04},
            "历史波动率基线": {"样本数": 10, "覆盖率": 0.80, "平均区间宽度": 0.10, "中心预测MAE": 0.04},
        }
        selection = select_interval_method(
            {"严格时点价格依据可用": True},
            {"是否值得替代波动率基线": False, "CQR区间候选": {"样本数": 10}},
            [window, window, window],
        )
        self.assertEqual(selection["方法"], "历史波动率风险范围")


if __name__ == "__main__":
    unittest.main()
