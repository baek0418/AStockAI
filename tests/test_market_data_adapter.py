"""标准行情数据适配层的范围、质量与可追溯性测试。"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_data_adapter import (
    STANDARD_COLUMNS,
    adapt_history_file,
    build_standard_market_data,
    run_market_data_adapter,
)


def make_history():
    return pd.DataFrame(
        {
            "日期": ["2025-01-03", "2025-01-02", "2025-01-03", "2025-01-13"],
            "开盘": [10, 9, 10.5, 11],
            "收盘": [10.5, 10, 11, 11.2],
            "最高": [11, 10.5, 11.5, 11.5],
            "最低": [9.5, 8.8, 10, 10.8],
            "成交量": [100, 120, 130, 140],
        }
    )


class MarketDataAdapterTests(unittest.TestCase):
    def test_adapt_keeps_last_duplicate_and_traceability_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            history_file = Path(directory) / "测试股票历史.csv"
            make_history().to_csv(history_file, index=False)
            dataset, quality = adapt_history_file(history_file, "测试股票", "600001")
            self.assertEqual(list(dataset.columns), STANDARD_COLUMNS)
            self.assertEqual(dataset["instrument"].unique().tolist(), ["sh600001"])
            self.assertEqual(dataset["date"].tolist(), ["2025-01-02", "2025-01-03", "2025-01-13"])
            self.assertEqual(dataset.loc[1, "close"], 11.0)
            self.assertEqual(quality["重复日期行数"], 1)
            self.assertEqual(quality["复权方式"], "unknown")
            self.assertEqual(quality["长日期间隔"][0]["自然日间隔"], 10)

    def test_build_ignores_nested_and_skips_non_history_quote_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "market").mkdir(parents=True)
            (data / "on_demand").mkdir()
            make_history().to_csv(data / "测试股票历史.csv", index=False)
            pd.DataFrame({"股票名称": ["测试股票"], "当前价格": [10]}).to_csv(data / "测试股票.csv", index=False)
            make_history().to_csv(data / "market" / "指数.csv", index=False)
            make_history().to_csv(data / "on_demand" / "按需.csv", index=False)
            dataset, audit = build_standard_market_data(data, root, {"测试股票": "000001"})
            self.assertEqual(dataset["instrument"].unique().tolist(), ["sz000001"])
            self.assertEqual(len(dataset), 3)
            self.assertEqual(len(audit), 2)
            self.assertEqual(
                next(item for item in audit if item["文件"] == "测试股票.csv")["状态"],
                "跳过",
            )

    def test_run_writes_only_research_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            make_history().to_csv(data / "测试股票历史.csv", index=False)
            dataset, audit, data_file, json_file, markdown_file = run_market_data_adapter(root)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(audit["标的数"], 1)
            self.assertTrue(data_file.is_file())
            self.assertTrue(json_file.is_file())
            self.assertTrue(markdown_file.is_file())
            self.assertFalse(any(data.glob("standard_market_data*.csv")))


if __name__ == "__main__":
    unittest.main()
