"""daily_signal 仅使用临时 JSON 快照的单元测试。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from daily_report import (
    AI_FALLBACK_TEXT,
    apply_daily_signal,
    create_watchlist_section,
    generate_ai_summary,
)
from daily_signal import run_daily_signal
from stock_analysis import build_ai_summary, create_markdown_content


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class DailySignalTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.output = self.directory / "output"
        self.output.mkdir()
        self.watchlist_file = self.directory / "watchlist.json"
        write_json(
            self.watchlist_file,
            {"stocks": [{"code": "000001", "name": "测试股票", "enable": True}]},
        )
        write_json(
            self.output / "quant_snapshot_2026-07-23.json",
            {
                "快照日期": "2026-07-23",
                "股票排行榜": [
                    {"股票代码": "000001", "股票名称": "测试股票", "综合评分": 60, "RSI": 55, "MA5": 9, "MA20": 10, "MACD": -0.1, "建议": "观望"}
                ],
            },
        )
        write_json(
            self.output / "quant_snapshot_2026-07-24.json",
            {
                "快照日期": "2026-07-24",
                "股票排行榜": [
                    {"股票代码": "000001", "股票名称": "测试股票", "综合评分": 70, "RSI": 60, "MA5": 11, "MA20": 10, "MACD": 0.2, "建议": "重点观察"}
                ],
            },
        )
        write_json(
            self.output / "watchlist_snapshot_2026-07-24.json",
            {"stocks": [{"code": "000001", "name": "测试股票", "trend": "均线多头", "risk": "正常"}]},
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_compares_current_snapshot_with_previous_trading_day(self):
        signal_file = run_daily_signal(self.output, self.watchlist_file)
        signal_data = json.loads(signal_file.read_text(encoding="utf-8"))
        stock = signal_data["stocks"][0]

        self.assertTrue(signal_data["前一交易日数据可用"])
        self.assertEqual(stock["当前指标"]["Score"], 70)
        self.assertEqual(stock["今日变化"]["Score变化"], 10)
        self.assertEqual(stock["今日变化"]["RSI变化"], 5)
        self.assertEqual(stock["今日变化"]["MA5/MA20关系变化"], "由MA5 低于 MA20变为MA5 高于 MA20")
        self.assertEqual(stock["今日变化"]["MACD状态变化"], "MACD 由负值转为正值")
        self.assertEqual(stock["信号分类"], "偏强")

    def test_missing_previous_snapshot_marks_changes_as_insufficient(self):
        (self.output / "quant_snapshot_2026-07-23.json").unlink()
        signal_file = run_daily_signal(self.output, self.watchlist_file)
        stock = json.loads(signal_file.read_text(encoding="utf-8"))["stocks"][0]

        self.assertIn("缺少前一交易日快照", stock["数据状态"])
        self.assertEqual(stock["今日变化"]["Score变化"], "数据不足")
        self.assertEqual(stock["今日变化"]["MACD状态变化"], "数据不足")

    def test_new_stock_uses_truncated_local_history_when_previous_snapshot_lacks_it(self):
        watchlist = json.loads(self.watchlist_file.read_text(encoding="utf-8"))
        watchlist["stocks"].append({"code": "000002", "name": "新纳入股票", "enable": True})
        write_json(self.watchlist_file, watchlist)
        latest = json.loads((self.output / "quant_snapshot_2026-07-24.json").read_text(encoding="utf-8"))
        latest["股票排行榜"].append(
            {"股票代码": "000002", "股票名称": "新纳入股票", "数据文件": "新纳入股票历史.csv", "综合评分": 65, "RSI": 55, "MA5": 12, "MA20": 11, "MACD": 0.2, "建议": "重点观察"}
        )
        write_json(self.output / "quant_snapshot_2026-07-24.json", latest)
        data_directory = self.directory / "data"
        data_directory.mkdir()
        dates = pd.bdate_range("2026-06-01", "2026-07-24")
        pd.DataFrame(
            {"日期": dates, "收盘": range(10, 10 + len(dates)), "成交量": [1000] * len(dates)}
        ).to_csv(data_directory / "新纳入股票历史.csv", index=False, encoding="utf-8-sig")

        signal_file = run_daily_signal(self.output, self.watchlist_file, data_directory)
        stocks = json.loads(signal_file.read_text(encoding="utf-8"))["stocks"]
        stock = next(item for item in stocks if item["股票名称"] == "新纳入股票")

        self.assertIn("本地日线回溯", stock["数据状态"])
        self.assertIsInstance(stock["今日变化"]["Score变化"], (int, float))

    def test_rule_daily_report_contains_signal_when_ai_is_unavailable(self):
        signal_file = run_daily_signal(self.output, self.watchlist_file)
        signal_data = json.loads(signal_file.read_text(encoding="utf-8"))
        watchlist_stocks = [
            {
                "股票代码": "000001",
                "股票名称": "测试股票",
                "别名": "",
                "优先级": 1,
                "标签": [],
                "持仓成本": None,
                "目标价": None,
                "备注": "",
                "摘要数据可用": False,
            }
        ]
        apply_daily_signal(watchlist_stocks, signal_data)
        previous_disable = os.environ.get("ASTOCKAI_DISABLE_AI")
        os.environ["ASTOCKAI_DISABLE_AI"] = "1"
        try:
            ai_summary = generate_ai_summary({}, watchlist_stocks, signal_data)
        finally:
            if previous_disable is None:
                os.environ.pop("ASTOCKAI_DISABLE_AI", None)
            else:
                os.environ["ASTOCKAI_DISABLE_AI"] = previous_disable

        self.assertEqual(ai_summary, AI_FALLBACK_TEXT)
        report_section = create_watchlist_section(watchlist_stocks)
        self.assertIn("今日变化", report_section)
        self.assertIn("观察重点", report_section)
        self.assertIn("Score：10", report_section)

    def test_ai_calls_use_shared_prompt_and_required_options(self):
        with patch("daily_report.call_ai_model", return_value="说明") as daily_call:
            result = generate_ai_summary({}, [], {"stocks": []})
        self.assertEqual(result, "说明")
        self.assertEqual(daily_call.call_args.kwargs["temperature"], 0.2)
        self.assertEqual(daily_call.call_args.kwargs["max_tokens"], 800)
        self.assertIn("量化研究", daily_call.call_args.kwargs["system_prompt"])

        facts = {"股票代码": "000001", "股票名称": "测试", "别名": "", "优先级": 1, "标签": [], "备注": "", "持仓成本": None, "目标价": None, "综合评分": 70, "建议": "重点观察", "趋势": "均线多头", "RSI": 60, "MA5": 11, "MA20": 10, "MACD": 0.2, "风险标签": "正常", "数据来源": "quant"}
        with patch("stock_analysis.call_ai_model", return_value="说明") as stock_call:
            self.assertEqual(build_ai_summary(facts), "说明")
        self.assertEqual(stock_call.call_args.kwargs["temperature"], 0.2)
        self.assertEqual(stock_call.call_args.kwargs["max_tokens"], 800)
        markdown = create_markdown_content(facts, "规则内容", "说明")
        self.assertIn("## 趋势与动量", markdown)
        self.assertIn("## 风险与观察", markdown)
        self.assertIn("## 综合结论", markdown)


if __name__ == "__main__":
    unittest.main()
