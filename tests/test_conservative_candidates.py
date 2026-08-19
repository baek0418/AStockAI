"""20 个交易日稳健研究候选的无前视、市场门槛与排序测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from conservative_candidates import build_conservative_candidates


def write_history(path, closes, volume=1000):
    dates = pd.bdate_range("2026-01-01", periods=len(closes))
    pd.DataFrame(
        {"日期": dates, "收盘": closes, "成交量": [volume] * len(closes)}
    ).to_csv(path, index=False, encoding="utf-8-sig")


class ConservativeCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data = self.root / "data"
        self.market = self.data / "market"
        self.market.mkdir(parents=True)
        self.as_of_date = pd.bdate_range("2026-01-01", periods=100)[-1].date().isoformat()
        self.stock_rankings = []
        for name, score in (("甲", 85), ("乙", 80), ("丙", 75), ("丁", 70)):
            # 温和上行：20 日上涨约 5%，MA5 高于 MA20，且日波动很低。
            closes = 10 + np.arange(100) * 0.025
            write_history(self.data / f"{name}历史.csv", closes)
            self.stock_rankings.append(
                {
                    "股票名称": name,
                    "股票代码": f"00000{len(self.stock_rankings) + 1}",
                    "数据文件": f"{name}历史.csv",
                    "综合评分": score,
                    "RSI": 55,
                    "MACD": 0.2,
                }
            )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_strong_market(self):
        write_history(self.market / "沪深300_sh000300.csv", 100 + np.arange(100) * 0.2)

    def test_returns_three_sorted_candidates_only_after_all_gates_pass(self):
        self.write_strong_market()
        # 第四只过热，不能因为需要凑足三只而放行。
        self.stock_rankings[3]["RSI"] = 75

        result = build_conservative_candidates(
            self.stock_rankings, self.data, self.market, self.as_of_date
        )

        self.assertTrue(result["市场环境"]["passed"])
        self.assertEqual([item["股票名称"] for item in result["候选股票"]], ["甲", "乙", "丙"])
        self.assertEqual([item["股票名称"] for item in result["20日研究推荐"]], ["甲", "乙", "丙"])
        self.assertEqual(result["持有期（交易日）"], 20)
        self.assertTrue(any(item["股票名称"] == "丁" for item in result["排除记录"]))

    def test_weak_market_returns_no_candidates_even_when_stocks_pass(self):
        closes = 100 + np.arange(100) * 0.2
        closes[-1] = closes[-20] * 0.95
        write_history(self.market / "沪深300_sh000300.csv", closes)

        result = build_conservative_candidates(
            self.stock_rankings, self.data, self.market, self.as_of_date
        )

        self.assertFalse(result["市场环境"]["passed"])
        self.assertEqual(result["候选股票"], [])
        self.assertEqual(len(result["20日研究推荐"]), 3)
        self.assertTrue(all("市场防守观察" in item["推荐状态"] for item in result["20日研究推荐"]))
        self.assertIn("今日未给出稳健研究候选", result["说明"])


if __name__ == "__main__":
    unittest.main()
