import unittest

import pandas as pd

from astock_core.portfolio.portfolio_backtest import PortfolioConfig, calculate_market_exposure, run_portfolio_backtest


def make_panel(rows):
    return pd.DataFrame(rows)


class PortfolioBacktestTests(unittest.TestCase):
    def test_selects_highest_signal_and_tracks_benchmark(self):
        panel = make_panel(
            [
                {"日期": "2025-01-01", "股票": "A", "收盘": 10, "信号": 0.9},
                {"日期": "2025-01-01", "股票": "B", "收盘": 10, "信号": 0.1},
                {"日期": "2025-01-02", "股票": "A", "收盘": 12, "信号": 0.9},
                {"日期": "2025-01-02", "股票": "B", "收盘": 9, "信号": 0.1},
            ]
        )
        benchmark = pd.DataFrame({"日期": ["2025-01-01", "2025-01-02"], "收盘": [100, 105]})
        config = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            max_weight=1,
            rebalance_interval=5,
            commission_rate=0,
            minimum_commission=0,
            sell_stamp_duty_rate=0,
            slippage_rate=0,
            lot_size=1,
        )
        nav, trades, statistics = run_portfolio_backtest(panel, config, benchmark)
        self.assertEqual(trades.iloc[0]["股票"], "A")
        self.assertEqual(trades.iloc[0]["方向"], "买入")
        self.assertAlmostEqual(nav.iloc[-1]["策略净值"], 1.2)
        self.assertAlmostEqual(nav.iloc[-1]["基准净值"], 1.05)
        self.assertAlmostEqual(statistics["超额累计收益率"], 0.15)

    def test_sell_restriction_keeps_existing_position(self):
        panel = make_panel(
            [
                {"日期": "2025-01-01", "股票": "A", "收盘": 10, "信号": 1.0},
                {"日期": "2025-01-01", "股票": "B", "收盘": 10, "信号": 0.0},
                {"日期": "2025-01-02", "股票": "A", "收盘": 11, "信号": 0.0, "可卖出": False},
                {"日期": "2025-01-02", "股票": "B", "收盘": 10, "信号": 1.0},
            ]
        )
        config = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            max_weight=1,
            rebalance_interval=1,
            commission_rate=0,
            minimum_commission=0,
            sell_stamp_duty_rate=0,
            slippage_rate=0,
            lot_size=1,
        )
        nav, trades, _ = run_portfolio_backtest(panel, config)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["股票"], "A")
        self.assertEqual(nav.iloc[-1]["持仓数量"], 1)

    def test_costs_reduce_net_value_and_minimum_commission_applies(self):
        panel = make_panel(
            [
                {"日期": "2025-01-01", "股票": "A", "收盘": 10, "信号": 1.0},
                {"日期": "2025-01-02", "股票": "A", "收盘": 10, "信号": 1.0},
            ]
        )
        config = PortfolioConfig(
            initial_capital=1000,
            max_positions=1,
            max_weight=1,
            rebalance_interval=5,
            commission_rate=0,
            minimum_commission=5,
            sell_stamp_duty_rate=0,
            slippage_rate=0,
            lot_size=1,
        )
        nav, trades, statistics = run_portfolio_backtest(panel, config)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades.iloc[0]["费用"], 5)
        self.assertLess(nav.iloc[-1]["策略净值"], 1)
        self.assertEqual(statistics["总费用"], 5.0)

    def test_rejects_duplicate_signal_rows(self):
        panel = make_panel(
            [
                {"日期": "2025-01-01", "股票": "A", "收盘": 10, "信号": 1.0},
                {"日期": "2025-01-01", "股票": "A", "收盘": 11, "信号": 0.5},
            ]
        )
        with self.assertRaisesRegex(ValueError, "只能有一条信号"):
            run_portfolio_backtest(panel)

    def test_explicit_rebalance_keeps_positions_during_signal_gaps(self):
        panel = make_panel(
            [
                {"日期": "2025-01-01", "股票": "A", "收盘": 10, "信号": 1.0, "调仓": True},
                {"日期": "2025-01-02", "股票": "A", "收盘": 11, "信号": None, "调仓": False},
                {"日期": "2025-01-03", "股票": "A", "收盘": 12, "信号": None, "调仓": False},
            ]
        )
        config = PortfolioConfig(
            initial_capital=1000, max_positions=1, max_weight=1, commission_rate=0,
            minimum_commission=0, sell_stamp_duty_rate=0, slippage_rate=0, lot_size=1,
        )
        nav, trades, _ = run_portfolio_backtest(panel, config)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(nav.iloc[-1]["策略净值"], 1.2)

    def test_market_trend_guard_exits_using_only_previous_benchmark_close(self):
        panel = make_panel(
            [
                {"日期": "2025-01-01", "股票": "A", "收盘": 10, "信号": 1.0},
                {"日期": "2025-01-02", "股票": "A", "收盘": 10, "信号": 1.0},
                {"日期": "2025-01-03", "股票": "A", "收盘": 10, "信号": 1.0},
            ]
        )
        benchmark = pd.DataFrame({"日期": ["2025-01-01", "2025-01-02", "2025-01-03"], "收盘": [100, 90, 80]})
        config = PortfolioConfig(
            initial_capital=1000, max_positions=1, max_weight=1, rebalance_interval=1,
            commission_rate=0, minimum_commission=0, sell_stamp_duty_rate=0, slippage_rate=0,
            lot_size=1, market_ma_window=2, risk_off_exposure=0,
        )
        nav, trades, _ = run_portfolio_backtest(panel, config, benchmark)
        self.assertEqual(nav.iloc[1]["目标暴露"], 1.0)
        self.assertEqual(nav.iloc[2]["目标暴露"], 0.0)
        self.assertEqual(trades.iloc[-1]["方向"], "卖出")

    def test_volatility_target_reduces_exposure_after_sufficient_history(self):
        history = pd.Series([100.0, 110.0, 100.0, 111.0])
        config = PortfolioConfig(target_annual_volatility=0.10, volatility_lookback_days=3)
        exposure, state = calculate_market_exposure(history, config)
        self.assertLess(exposure, 1.0)
        self.assertIn("波动率目标", state)


if __name__ == "__main__":
    unittest.main()
