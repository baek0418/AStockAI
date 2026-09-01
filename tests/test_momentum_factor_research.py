import unittest

import pandas as pd

from momentum_factor_research import build_cross_sectional_momentum_signals, calculate_temporal_windows
from portfolio_backtest import PortfolioConfig


class MomentumFactorResearchTests(unittest.TestCase):
    def test_12_1_momentum_uses_only_prices_available_on_signal_date(self):
        dates = pd.bdate_range("2024-01-01", periods=280)
        base = pd.DataFrame(
            [
                {"日期": date, "股票名称": name, "收盘": close}
                for name, multiplier in (("A", 1.0), ("B", 1.5))
                for index, date in enumerate(dates)
                for close in [10 + index * multiplier]
            ]
        )
        original = build_cross_sectional_momentum_signals(
            base, lookback_days=252, skip_recent_days=21, min_cross_section=2
        )
        changed = base.copy()
        changed.loc[changed["日期"] > dates[-2], "收盘"] *= 100
        updated = build_cross_sectional_momentum_signals(
            changed, lookback_days=252, skip_recent_days=21, min_cross_section=2
        )
        pd.testing.assert_frame_equal(
            original[original["日期"] < dates[-1]].reset_index(drop=True),
            updated[updated["日期"] < dates[-1]].reset_index(drop=True),
        )

    def test_signal_ranks_cross_section_and_requires_coverage(self):
        dates = pd.bdate_range("2024-01-01", periods=280)
        frame = pd.DataFrame(
            [
                {"日期": date, "股票名称": name, "收盘": 10 + index * multiplier}
                for name, multiplier in (("A", 1.0), ("B", 2.0), ("C", 3.0))
                for index, date in enumerate(dates)
            ]
        )
        signals = build_cross_sectional_momentum_signals(
            frame, lookback_days=252, skip_recent_days=21, min_cross_section=3
        )
        last = signals[signals["日期"] == signals["日期"].max()].set_index("股票名称")
        self.assertGreater(last.loc["C", "信号"], last.loc["B", "信号"])
        self.assertGreater(last.loc["B", "信号"], last.loc["A", "信号"])
        with self.assertRaisesRegex(ValueError, "横截面覆盖"):
            build_cross_sectional_momentum_signals(
                frame[frame["股票名称"] != "C"], lookback_days=252, skip_recent_days=21, min_cross_section=3
            )

    def test_missing_stock_day_does_not_shift_momentum_lookback_endpoint(self):
        dates = pd.bdate_range("2024-01-01", periods=280)
        frame = pd.DataFrame(
            [
                {"日期": date, "股票名称": name, "收盘": 10 + index * multiplier}
                for name, multiplier in (("A", 1.0), ("B", 2.0))
                for index, date in enumerate(dates)
            ]
        )
        # A 少了一个中间交易日；计算时仍须按全市场日期 t-252、t-21 取端点。
        frame = frame[~((frame["股票名称"] == "A") & (frame["日期"] == dates[10]))]
        signals = build_cross_sectional_momentum_signals(
            frame, lookback_days=252, skip_recent_days=21, min_cross_section=1
        )
        signal_date = dates[260]
        actual = signals.loc[
            (signals["日期"] == signal_date) & (signals["股票名称"] == "A"), "12-1月动量"
        ].iloc[0]
        expected = (10 + 239) / (10 + 8) - 1
        self.assertAlmostEqual(actual, expected)

    def test_temporal_windows_start_on_rebalance_dates(self):
        dates = pd.bdate_range("2025-01-01", periods=9)
        panel = pd.DataFrame(
            [
                {"日期": date, "股票": "A", "收盘": 10.0, "信号": 1.0, "调仓": index in (0, 3, 6)}
                for index, date in enumerate(dates)
            ]
        )
        benchmark = pd.DataFrame({"日期": dates, "收盘": range(100, 109)})
        config = PortfolioConfig(
            initial_capital=1000, max_positions=1, max_weight=1, commission_rate=0,
            minimum_commission=0, sell_stamp_duty_rate=0, slippage_rate=0, lot_size=1,
        )
        windows = calculate_temporal_windows(panel, benchmark, config)
        self.assertEqual(len(windows), 3)
        self.assertEqual([window["日期范围"][0] for window in windows], ["2025-01-01", "2025-01-06", "2025-01-09"])
