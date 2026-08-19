"""v5.1 基准感知研究的目录隔离、无泄漏与安全拒绝测试。"""

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from prediction_benchmark_data import (
    BENCHMARKS,
    extract_index_day_rows,
    normalize_index_dataframe,
    download_benchmarks,
)
from prediction_data_audit import audit_data
from prediction_features import HORIZON_DAYS
from prediction_features_v2 import (
    FEATURE_COLUMNS_V2,
    LABEL_COLUMN_V2,
    build_feature_dataset_v2,
    get_labeled_dataset_v2,
)
from prediction_model import calculate_probability_bins, evaluate_rolling_windows
from predict_outperformance import predict_outperformance
from update_data import get_stock


def make_history(days=100, start="2025-01-01", base=10):
    dates = pd.bdate_range(start, periods=days)
    indexes = np.arange(days)
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": base + indexes * 0.05,
            "收盘": base + indexes * 0.05 + np.sin(indexes / 3),
            "最高": base + indexes * 0.05 + 1,
            "最低": base + indexes * 0.05 - 1,
            "成交量": 100000 + indexes * 100,
        }
    )


class PredictionV51Tests(unittest.TestCase):
    def write_dataset(self, root, stock_history=None, benchmark_history=None):
        data = root / "data"
        market = data / "market"
        market.mkdir(parents=True, exist_ok=True)
        (stock_history if stock_history is not None else make_history()).to_csv(
            data / "测试股票历史.csv", index=False
        )
        (benchmark_history if benchmark_history is not None else make_history(base=3000)).to_csv(
            market / BENCHMARKS["沪深300"]["file_name"], index=False
        )

    def test_v2_features_do_not_change_when_future_rows_are_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock = make_history(100)
            benchmark = make_history(100, base=3000)
            self.write_dataset(root, stock, benchmark)
            original, _ = build_feature_dataset_v2(project_directory=root)
            stock_extended = pd.concat([stock, make_history(10, "2025-06-01")], ignore_index=True)
            benchmark_extended = pd.concat(
                [benchmark, make_history(10, "2025-06-01", base=3000)], ignore_index=True
            )
            self.write_dataset(root, stock_extended, benchmark_extended)
            extended, _ = build_feature_dataset_v2(project_directory=root)
            pd.testing.assert_frame_equal(
                original[FEATURE_COLUMNS_V2].iloc[:90].reset_index(drop=True),
                extended[FEATURE_COLUMNS_V2].iloc[:90].reset_index(drop=True),
            )

    def test_last_five_rows_have_no_outperformance_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_dataset(root)
            features, _ = build_feature_dataset_v2(project_directory=root)
            self.assertTrue(features[LABEL_COLUMN_V2].tail(HORIZON_DAYS).isna().all())
            self.assertEqual(len(get_labeled_dataset_v2(features)), len(features) - HORIZON_DAYS - 59)

    def test_missing_benchmark_safely_refuses_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            make_history().to_csv(data / "测试股票历史.csv", index=False)
            with self.assertRaisesRegex(ValueError, "缺少市场基准数据"):
                build_feature_dataset_v2(project_directory=root)
            self.assertFalse((data / "market" / BENCHMARKS["沪深300"]["file_name"]).exists())

    def test_benchmark_download_is_isolated_and_failure_writes_no_empty_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            market = Path(directory) / "data" / "market"

            def fake_index_daily(_):
                return make_history(10).values.tolist()

            result = download_benchmarks(["沪深300"], market, fake_index_daily)[0]
            self.assertEqual(result["status"], "success")
            self.assertTrue((market / BENCHMARKS["沪深300"]["file_name"]).is_file())
            metadata_file = market / "沪深300_sh000300.metadata.json"
            self.assertTrue(metadata_file.is_file())
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            self.assertEqual(metadata["data_type"], "index_daily")
            self.assertEqual(metadata["adjustment"], "none")
            self.assertEqual(metadata["source"], "腾讯财经")
            self.assertFalse((market.parent / BENCHMARKS["沪深300"]["file_name"]).exists())

            def failed_index_daily(_):
                raise ConnectionError("network failed")

            failed = download_benchmarks(["中证1000"], market, failed_index_daily)[0]
            self.assertEqual(failed["status"], "failed")
            self.assertFalse((market / BENCHMARKS["中证1000"]["file_name"]).exists())

    def test_index_day_is_preferred_and_qfqday_is_not_required(self):
        response = {
            "data": {
                "sh000300": {
                    "day": [["2025-01-02", "3900", "3920", "3930", "3890", "123456"]],
                    "qfqday": [],
                }
            }
        }
        rows = extract_index_day_rows(response, "sh000300")
        frame = normalize_index_dataframe(rows)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "收盘"], 3920.0)

    def test_missing_or_empty_index_day_fails_without_csv(self):
        with self.assertRaisesRegex(ValueError, "缺少 day"):
            extract_index_day_rows({"data": {"sh000300": {"qfqday": [["x"]]}}}, "sh000300")
        with self.assertRaisesRegex(ValueError, "day 日线为空"):
            extract_index_day_rows({"data": {"sh000300": {"day": []}}}, "sh000300")
        with tempfile.TemporaryDirectory() as directory:
            market = Path(directory) / "data" / "market"
            failed = download_benchmarks(
                ["沪深300"], market, lambda _: [["2025-01-02", "1", "2"]]
            )[0]
            self.assertEqual(failed["status"], "failed")
            self.assertFalse((market / BENCHMARKS["沪深300"]["file_name"]).exists())

    def test_stock_qfqday_parser_is_unchanged(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"sz000001": {"qfqday": [["2025-01-02", "1", "2", "3", "0.5", "100"]]}}}

        with patch("update_data.requests.get", return_value=FakeResponse()):
            rows = get_stock("sz000001")
        self.assertEqual(rows[0][2], "2")

    def test_stock_day_fallback_is_used_when_qfqday_is_unavailable(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"sh601399": {"qfqday": [], "day": [["2025-01-02", "1", "2", "3", "0.5", "100"]]}}}

        with patch("update_data.requests.get", return_value=FakeResponse()):
            rows = get_stock("sh601399")
        self.assertEqual(rows[0][2], "2")

    def test_v2_evaluation_accepts_target_wording_and_uses_custom_calibration_key(self):
        bins = calculate_probability_bins(
            [0, 1], [0.45, 0.55], actual_positive_rate_key="实际跑赢基准率"
        )
        self.assertIn("实际跑赢基准率", bins[1])
        self.assertNotIn("实际上涨率", bins[1])
        short_dataset = pd.DataFrame({"日期": pd.bdate_range("2025-01-01", periods=10)})
        result = evaluate_rolling_windows(
            short_dataset,
            outcome_description="跑赢市场基准",
            actual_positive_rate_key="实际跑赢基准率",
        )
        self.assertFalse(result["ready"])

    def test_audit_counts_only_trainable_root_csv_and_ignores_market(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "market").mkdir(parents=True)
            make_history().to_csv(data / "测试股票历史.csv", index=False)
            make_history(1).to_csv(data / "一行.csv", index=False)
            make_history().to_csv(data / "market" / BENCHMARKS["沪深300"]["file_name"], index=False)
            audit = audit_data(data, root)
            self.assertEqual(audit["股票数量"], 1)
            self.assertIn("当前仅有 16 只股票", audit["横截面限制"])

    def test_prediction_refuses_when_no_validated_v51_model(self):
        with tempfile.TemporaryDirectory() as directory:
            result = predict_outperformance("300750", Path(directory))
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("未找到 v5.1 已训练模型", result["message"])


if __name__ == "__main__":
    unittest.main()
