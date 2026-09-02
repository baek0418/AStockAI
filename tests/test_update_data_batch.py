import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from astock_core.data.market_data_sources import MarketDataFetchResult
from astock_core.data.update_data import run_update_data


def make_result(source="主源", fallback=False):
    history = pd.DataFrame(
        {
            "日期": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 11.0],
            "最高": [10.7, 11.2],
            "最低": [9.8, 10.2],
            "成交量": [1000, 1200],
        }
    )
    return MarketDataFetchResult(
        history=history,
        source=source,
        adjustment="qfq",
        used_fallback=fallback,
        attempted_sources=("主源", source) if fallback else (source,),
        failures=({"数据源": "主源", "原因": "超时"},) if fallback else (),
        request_attempts=2 if fallback else 1,
    )


class UpdateDataBatchTests(unittest.TestCase):
    def test_batch_update_keeps_per_stock_source_provenance_and_partial_status(self):
        stocks = [
            {"code": "000001", "name": "甲", "market_code": "sz000001"},
            {"code": "600000", "name": "乙", "market_code": "sh600000"},
            {"code": "000002", "name": "失败股", "market_code": "sz000002"},
        ]

        def fetch_history(market_code):
            if market_code == "sz000002":
                raise ValueError("所有来源均失败")
            return make_result("备用源" if market_code == "sh600000" else "主源", market_code == "sh600000")

        with tempfile.TemporaryDirectory() as directory:
            result = run_update_data(Path(directory), stocks, max_workers=2, fetch_history=fetch_history)
            data = Path(directory) / "data"
            provenance = json.loads((data / "provenance" / "600000.json").read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["details"]["success_count"], 2)
            self.assertEqual(result["details"]["failed_count"], 1)
            self.assertEqual(result["details"]["fallback_count"], 1)
            self.assertTrue((data / "甲历史.csv").is_file())
            self.assertFalse((data / "失败股历史.csv").exists())
            self.assertEqual(provenance["数据源"], "备用源")
            self.assertTrue(provenance["是否使用备用源"])
            self.assertEqual(provenance["请求尝试次数"], 2)
            self.assertTrue(Path(result["details"]["event_log"]).is_file())


if __name__ == "__main__":
    unittest.main()
