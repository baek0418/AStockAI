import unittest

import pandas as pd

from expert_research import build_expert_research_memo, build_price_research_evidence


class ExpertResearchTests(unittest.TestCase):
    def test_price_evidence_uses_local_history_and_aligned_benchmark(self):
        dates = pd.bdate_range("2025-01-01", periods=270)
        history = pd.DataFrame(
            {
                "日期": dates,
                "收盘": [100 + index for index in range(270)],
                "成交量": [1000 + index * 10 for index in range(270)],
            }
        )
        benchmark = pd.DataFrame(
            {"日期": dates, "收盘": [100 + index * 0.5 for index in range(270)], "成交量": 1000}
        )
        evidence = build_price_research_evidence(history, benchmark)

        self.assertEqual(evidence["数据状态"], "可用")
        self.assertAlmostEqual(evidence["区间收益率"]["20日"], round((369 / 349 - 1) * 100, 2))
        expected_relative = ((369 / 349 - 1) - (234.5 / 224.5 - 1)) * 100
        self.assertAlmostEqual(evidence["相对沪深300"]["20日"], round(expected_relative, 2))
        self.assertEqual(evidence["价格位置"]["区间位置"], 100.0)
        self.assertGreater(evidence["成交活跃度"], 1)

    def test_memo_surfaces_fundamental_gap_and_counter_evidence(self):
        stock_evidence = {"当前量化证据": {"MA5": 11, "MA20": 10, "MACD": 0.2, "RSI": 72}}
        price_evidence = {
            "数据状态": "可用",
            "区间收益率": {"20日": 8.0},
            "相对沪深300": {"20日": 3.0},
            "价格位置": {"观察窗口交易日": 252, "距窗口高点": -1.0},
            "成交活跃度": 1.3,
        }
        memo = build_expert_research_memo(stock_evidence, price_evidence)

        self.assertIn("同向偏强", memo["核心研究论点"])
        self.assertTrue(any("RSI" in item for item in memo["相反证据与风险"]))
        self.assertIn("未接入", memo["基本面与行业证据"])


if __name__ == "__main__":
    unittest.main()
