import unittest

import pandas as pd

from astock_core.research.oos_portfolio_research import PROBABILITY_COLUMN, create_execution_signal_panel


class OosPortfolioResearchTests(unittest.TestCase):
    def test_signals_execute_on_next_trading_day_and_keep_gap_prices(self):
        features = pd.DataFrame(
            [
                {"日期": "2025-01-01", "股票名称": "A", "收盘": 10},
                {"日期": "2025-01-02", "股票名称": "A", "收盘": 11},
                {"日期": "2025-01-03", "股票名称": "A", "收盘": 12},
                {"日期": "2025-01-01", "股票名称": "B", "收盘": 20},
                {"日期": "2025-01-02", "股票名称": "B", "收盘": 21},
                {"日期": "2025-01-03", "股票名称": "B", "收盘": 22},
            ]
        )
        signals = [
            pd.DataFrame(
                {
                    "日期": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                    "股票名称": ["A", "B"],
                    PROBABILITY_COLUMN: [0.8, 0.2],
                }
            )
        ]
        panel = create_execution_signal_panel(features, signals)
        self.assertEqual(panel["日期"].dt.strftime("%Y-%m-%d").unique().tolist(), ["2025-01-02"])
        self.assertTrue(panel["调仓"].all())
        self.assertEqual(panel.sort_values("股票")["信号"].tolist(), [0.8, 0.2])

    def test_rejects_when_no_stock_has_complete_execution_coverage(self):
        features = pd.DataFrame(
            [{"日期": "2025-01-01", "股票名称": "A", "收盘": 10}, {"日期": "2025-01-02", "股票名称": "B", "收盘": 20}]
        )
        signals = [pd.DataFrame({"日期": [pd.Timestamp("2025-01-01")], "股票名称": ["A"], PROBABILITY_COLUMN: [0.7]})]
        with self.assertRaisesRegex(ValueError, "没有股票覆盖完整"):
            create_execution_signal_panel(features, signals)

    def test_rebalance_interval_marks_only_every_nth_signal_day(self):
        features = pd.DataFrame(
            [
                {"日期": date, "股票名称": "A", "收盘": 10 + index}
                for index, date in enumerate(pd.bdate_range("2025-01-01", periods=5))
            ]
        )
        signals = [
            pd.DataFrame(
                {
                    "日期": pd.bdate_range("2025-01-01", periods=4),
                    "股票名称": ["A"] * 4,
                    PROBABILITY_COLUMN: [0.7] * 4,
                }
            )
        ]
        panel = create_execution_signal_panel(features, signals, rebalance_interval=2)
        marked_dates = panel.loc[panel["调仓"], "日期"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(marked_dates, ["2025-01-02", "2025-01-06"])


if __name__ == "__main__":
    unittest.main()
