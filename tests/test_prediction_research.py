"""未来 5 日上涨概率研究的无泄漏与时间切分测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from predict_probability import predict_probability
from prediction_features import (
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    build_feature_dataset,
    build_stock_feature_frame,
)
from prediction_model import GAP_DAYS, create_rolling_windows, split_by_window


def make_history(days=100, start="2025-01-01"):
    dates = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {
            "日期": dates,
            "收盘": 10 + np.arange(days) * 0.05 + np.sin(np.arange(days) / 3),
            "成交量": 100000 + np.arange(days) * 100,
        }
    )


class PredictionFeatureTests(unittest.TestCase):
    def test_future_rows_do_not_change_historical_features(self):
        history = make_history(100)
        original = build_stock_feature_frame(history, "测试股", "000001")
        extended_history = pd.concat([history, make_history(10, "2025-06-01")], ignore_index=True)
        extended = build_stock_feature_frame(extended_history, "测试股", "000001")

        original_features = original[FEATURE_COLUMNS].iloc[:90].reset_index(drop=True)
        extended_features = extended[FEATURE_COLUMNS].iloc[:90].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_features, extended_features)

    def test_label_uses_exactly_five_future_trading_rows_and_last_rows_excluded(self):
        history = make_history(100)
        features = build_stock_feature_frame(history, "测试股", "000001")
        expected_return = history["收盘"].iloc[20 + HORIZON_DAYS] / history["收盘"].iloc[20] - 1
        self.assertAlmostEqual(features["future_5d_return"].iloc[20], expected_return)
        self.assertTrue(features["target_up_5d"].tail(HORIZON_DAYS).isna().all())

    def test_data_root_scan_does_not_read_on_demand_subdirectory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_directory = root / "data"
            on_demand_directory = data_directory / "on_demand"
            on_demand_directory.mkdir(parents=True)
            make_history(100).to_csv(data_directory / "正式股票历史.csv", index=False)
            make_history(100).to_csv(on_demand_directory / "临时股票历史.csv", index=False)

            dataset, skipped = build_feature_dataset(data_directory=data_directory, project_directory=root)
            self.assertEqual(dataset["股票名称"].unique().tolist(), ["正式股票"])
            self.assertEqual(skipped, [])

    def test_explicit_research_pool_filter_excludes_non_member_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_directory = root / "data"
            data_directory.mkdir()
            make_history(100).to_csv(data_directory / "研究池内历史.csv", index=False)
            make_history(100).to_csv(data_directory / "研究池外历史.csv", index=False)

            with patch(
                "prediction_features.create_stock_code_lookup",
                return_value={"研究池内": "000001", "研究池外": "000002"},
            ):
                dataset, skipped = build_feature_dataset(
                    data_directory=data_directory,
                    project_directory=root,
                    allowed_stock_codes={"000001"},
                )

            self.assertEqual(dataset["股票名称"].unique().tolist(), ["研究池内"])
            self.assertEqual(skipped, [{"file": "研究池外历史.csv", "reason": "不在当前启用的研究股票池快照中。"}])

    def test_rolling_windows_keep_dates_whole_and_insert_gap(self):
        dates = pd.bdate_range("2024-01-01", periods=120)
        dataset = pd.DataFrame(
            [
                {"日期": date, "股票名称": name, "target_up_5d": index % 2}
                for index, date in enumerate(dates)
                for name in ("甲", "乙")
            ]
        )
        windows = create_rolling_windows(dataset["日期"])

        self.assertGreaterEqual(len(windows), 3)
        for window in windows:
            train, test = split_by_window(dataset, window)
            self.assertEqual(len(window.gap_dates), GAP_DAYS)
            self.assertTrue(set(window.train_dates).isdisjoint(window.test_dates))
            self.assertTrue(set(window.gap_dates).isdisjoint(window.train_dates))
            self.assertTrue(set(window.gap_dates).isdisjoint(window.test_dates))
            self.assertEqual(set(train["日期"]), set(window.train_dates))
            self.assertEqual(set(test["日期"]), set(window.test_dates))

    def test_prediction_without_validated_model_refuses_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = predict_probability("300058", project_directory=Path(temporary_directory))
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("未找到已训练模型", result["message"])


if __name__ == "__main__":
    unittest.main()
