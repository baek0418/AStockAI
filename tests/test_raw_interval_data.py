"""原始价格区间快照测试：只使用构造行情，不访问网络。"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from astock_core.data.market_data_sources import MarketDataFetchResult
from astock_core.data.raw_interval_data import snapshot_raw_interval_data


def make_result(market_code, retry_policy=None, pacer=None):
    del retry_policy, pacer
    history = pd.DataFrame(
        {
            "日期": pd.to_datetime(["2026-08-31", "2026-09-01"]), "开盘": [10, 11], "收盘": [11, 12],
            "最高": [12, 13], "最低": [9, 10], "成交量": [100, 200],
        }
    )
    return MarketDataFetchResult(
        history=history, source="测试 raw", adjustment="raw", used_fallback=False,
        attempted_sources=("测试 raw",), failures=(), request_attempts=1,
    )


class RawIntervalDataTests(unittest.TestCase):
    def test_snapshot_writes_isolated_raw_history_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = snapshot_raw_interval_data(
                root,
                stock_universe=[{"code": "000001", "name": "测试股", "market_code": "sz000001"}],
                max_workers=1,
                request_interval_seconds=0,
                fetch_history=make_result,
            )
            data_directory = root / "data" / "raw_interval"
            provenance = json.loads((data_directory / "provenance" / "000001.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "success")
            self.assertTrue((data_directory / "测试股历史.csv").is_file())
            self.assertEqual(provenance["复权方式"], "raw")
            self.assertIn("区间研究专用", provenance["用途"])

    def test_explicit_empty_universe_does_not_fall_back_to_default_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            result = snapshot_raw_interval_data(Path(directory), stock_universe=[])
        self.assertEqual(result["status"], "failed")
        self.assertIn("股票池为空", result["message"])


if __name__ == "__main__":
    unittest.main()
