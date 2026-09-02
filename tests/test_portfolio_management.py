"""本地持仓账本测试：不读取真实持仓，也不访问网络。"""

import tempfile
import unittest
from pathlib import Path

from astock_core.portfolio.portfolio_management import (
    build_investment_review,
    build_portfolio_rows,
    load_portfolio,
    remove_holding,
    save_portfolio,
    summarize_portfolio,
    upsert_cash,
    upsert_holding,
)


class PortfolioManagementTests(unittest.TestCase):
    def test_missing_file_is_an_empty_local_portfolio(self):
        with tempfile.TemporaryDirectory() as directory:
            portfolio = load_portfolio(Path(directory) / "portfolio.json")
        self.assertEqual(portfolio["holdings"], [])
        self.assertEqual(portfolio["cash"], [])

    def test_save_reload_and_upsert_keep_account_data_local(self):
        with tempfile.TemporaryDirectory() as directory:
            portfolio_file = Path(directory) / "private" / "portfolio.json"
            portfolio = load_portfolio(portfolio_file)
            portfolio = upsert_holding(
                portfolio,
                {"account": "测试账户", "code": "600839", "name": "四川长虹", "quantity": 2000, "cost_price": 8.331, "category": "观察"},
            )
            portfolio = upsert_cash(portfolio, "测试账户", 1200)
            save_portfolio(portfolio_file, portfolio)
            loaded = load_portfolio(portfolio_file)

        self.assertEqual(loaded["holdings"][0]["quantity"], 2000)
        self.assertEqual(loaded["cash"], [{"account": "测试账户", "amount": 1200.0}])

    def test_local_quotes_drive_market_value_without_claiming_realtime_prices(self):
        portfolio = upsert_holding(
            {"version": "1.0", "holdings": [], "cash": []},
            {"account": "测试账户", "code": "600839", "name": "四川长虹", "quantity": 2000, "cost_price": 8.0},
        )
        rows = build_portfolio_rows(
            portfolio,
            {"600839": {"close": 7.12, "date": "2026-09-01", "advice": "重点观察"}},
            [{"股票代码": "600839", "信号分类": "偏强"}],
        )
        summary = summarize_portfolio(rows, portfolio)

        self.assertEqual(rows[0]["行情日期"], "2026-09-01")
        self.assertEqual(rows[0]["当前市值"], 14240.0)
        self.assertEqual(rows[0]["浮盈亏"], -1760.0)
        self.assertEqual(rows[0]["研究观察"], "偏强")
        self.assertEqual(summary["缺少本地报价数"], 0)

    def test_remove_requires_an_explicit_existing_position(self):
        portfolio = upsert_holding(
            {"version": "1.0", "holdings": [], "cash": []},
            {"account": "测试账户", "code": "600839", "name": "四川长虹", "quantity": 100, "cost_price": 8},
        )
        portfolio = remove_holding(portfolio, "测试账户", "600839")
        self.assertEqual(portfolio["holdings"], [])
        with self.assertRaisesRegex(ValueError, "未找到"):
            remove_holding(portfolio, "测试账户", "600839")

    def test_investment_review_prioritizes_missing_quotes_then_weak_research(self):
        rows = [
            {"账户": "测试", "股票代码": "600839", "股票名称": "甲", "当前市值": 1000, "研究观察": "偏强"},
            {"账户": "测试", "股票代码": "000001", "股票名称": "乙", "当前市值": 500, "研究观察": "偏弱"},
            {"账户": "测试", "股票代码": "000002", "股票名称": "丙", "当前市值": None, "研究观察": "数据不足"},
        ]
        review = build_investment_review(
            rows,
            [
                {"股票代码": "600839", "信号分类": "偏强", "观察重点": ["观察趋势是否延续。"], "当前指标": {"风险标签": "正常"}},
                {"股票代码": "000001", "信号分类": "偏弱", "观察重点": ["核对价格是否继续走弱。"], "当前指标": {"风险标签": "正常"}},
            ],
        )

        self.assertEqual([item["股票代码"] for item in review], ["000002", "000001", "600839"])
        self.assertIn("未匹配本地量化快照", review[0]["今日待核对"])
        self.assertIn("同日研究信号偏弱", review[1]["今日待核对"])
        self.assertEqual(review[2]["持仓占比"], 66.67)


if __name__ == "__main__":
    unittest.main()
