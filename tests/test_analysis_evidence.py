"""v4.6 证据层只读边界、市场事实与报告降级测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from astock_core.analysis.analysis_evidence import (
    build_report_evidence,
    load_market_context,
    load_quote_provenance_context,
)
from astock_core.reporting.daily_report import create_evidence_report_content, create_rule_cross_signal_summary
from astock_core.analysis.stock_analysis import create_evidence_markdown_content


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def write_market_csv(path, closes):
    dates = pd.bdate_range("2026-06-01", periods=len(closes))
    pd.DataFrame(
        {
            "日期": dates.strftime("%Y-%m-%d"),
            "开盘": closes,
            "收盘": closes,
            "最高": [value + 1 for value in closes],
            "最低": [value - 1 for value in closes],
            "成交量": [100] * len(closes),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


class AnalysisEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output = self.root / "output"
        self.market = self.root / "data" / "market"
        self.fundamentals = self.root / "data" / "fundamentals"
        self.output.mkdir(parents=True)
        self.market.mkdir(parents=True)
        self.fundamentals.mkdir(parents=True)
        self.watchlist = self.root / "watchlist.json"
        self.portfolio = self.root / "data" / "portfolio.json"
        write_json(self.watchlist, {"stocks": [{"code": "000001", "name": "测试股", "priority": 5, "enable": True}]})
        write_json(
            self.portfolio,
            {"version": "1.0", "holdings": [{
                "account": "本地账户", "code": "000001", "name": "测试股",
                "quantity": 100, "cost_price": 10.0, "category": "",
            }], "cash": []},
        )
        write_json(
            self.output / "quant_snapshot_2026-07-24.json",
            {"快照日期": "2026-07-24", "股票排行榜": [{
                "股票代码": "000001", "股票名称": "测试股", "综合评分": 70,
                "收盘价": 12.34, "RSI": 55, "MA5": 11, "MA20": 10, "MACD": 0.2, "建议": "重点观察",
                "扩展技术指标": {
                    "momentum_5d": 0.03, "momentum_10d": 0.05, "momentum_20d": 0.08,
                    "volatility_20d": 0.02, "volume_relative_5d": 1.3,
                },
            }]},
        )
        write_json(
            self.output / "daily_signal_2026-07-24.json",
            {"快照日期": "2026-07-24", "前一交易日数据可用": True, "stocks": [{
                "股票代码": "000001", "股票名称": "测试股", "数据状态": "前一交易日快照可用。",
                "当前指标": {"Score": 70, "RSI": 55, "MA5": 11, "MA20": 10, "MACD": 0.2, "趋势": "均线多头", "建议": "重点观察", "风险标签": "正常"},
                "今日变化": {"Score变化": 12, "RSI变化": 2, "MA5/MA20关系变化": "维持MA5 高于 MA20", "MACD状态变化": "MACD 正值扩大"},
                "信号分类": "偏强", "观察重点": ["观察 MA5 是否继续高于 MA20。"],
            }]},
        )
        write_json(self.output / "watchlist_snapshot_2026-07-24.json", {"stocks": []})
        write_json(
            self.fundamentals / "000001_fundamentals.json",
            {
                "数据状态": "可用", "股票代码": "000001",
                "最新报告": {"报告期": "2026-06-30", "公告日期": "2026-08-20", "指标": {
                    "营业总收入同比增长": 10.0, "归母净利润同比增长": 8.0,
                    "净资产收益率(加权)": 9.0, "资产负债率": 40.0,
                    "每股净资产": 10.0, "每股收益(基本)": 1.0,
                }},
                "公司与行业画像": {"数据状态": "可用", "字段": {
                    "所属行业": "测试行业", "主营业务": "测试主营业务",
                }},
                "来源": "测试来源", "官方核验页": "https://www.cninfo.com.cn/",
            },
        )
        write_market_csv(self.market / "沪深300_sh000300.csv", list(range(100, 125)))
        write_market_csv(self.market / "中证1000_sh000852.csv", list(range(200, 225)))

    def write_provenance(self, end_date="2026-07-24", fallback=False, attempts=1):
        provenance_directory = self.root / "data" / "provenance"
        provenance_directory.mkdir(parents=True, exist_ok=True)
        write_json(
            provenance_directory / "000001.json",
            {
                "数据源": "腾讯行情 qfqday" if not fallback else "东方财富 qfq kline",
                "复权方式": "qfq",
                "日期范围": ["2026-01-01", end_date],
                "是否使用备用源": fallback,
                "请求尝试次数": attempts,
                "更新时间": "2026-07-24 18:00:00",
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_market_context_uses_existing_index_csv_facts(self):
        market = load_market_context(self.market)
        csi300 = market["指数"]["沪深300"]
        self.assertEqual(csi300["数据状态"], "可用")
        self.assertEqual(csi300["数据截至日期"], "2026-07-03")
        self.assertAlmostEqual(csi300["1日涨跌"], round((124 / 123 - 1) * 100, 2))
        self.assertTrue(csi300["位于20日均线之上"])

    def test_missing_market_file_is_explicit_and_does_not_block_evidence(self):
        (self.market / "中证1000_sh000852.csv").unlink()
        evidence = build_report_evidence(self.output, self.market, self.watchlist, self.portfolio)
        self.assertIn("数据不足", evidence["市场环境"]["指数"]["中证1000"]["数据状态"])
        self.assertEqual(evidence["关注股票"][0]["当前量化证据"]["Score"], 70)

    def test_quote_provenance_distinguishes_same_day_fallback_and_missing_records(self):
        self.write_provenance(fallback=True, attempts=2)
        audit = load_quote_provenance_context(
            self.root / "data",
            [{"code": "000001", "name": "测试股"}, {"code": "000002", "name": "未更新股"}],
            "2026-07-24",
        )
        self.assertEqual(audit["可核对数"], 1)
        self.assertEqual(audit["未记录数"], 1)
        self.assertEqual(audit["备用源更新数"], 1)
        self.assertEqual(audit["重试后成功数"], 1)
        self.assertIn("审计不完整", audit["状态"])

    def test_stock_evidence_values_are_traceable_and_reports_exclude_probabilities(self):
        self.write_provenance(fallback=True, attempts=2)
        evidence = build_report_evidence(self.output, self.market, self.watchlist)
        stock = evidence["关注股票"][0]
        self.assertEqual(stock["当前量化证据"]["RSI"], 55)
        self.assertEqual(stock["当前量化证据"]["收盘价"], 12.34)
        self.assertEqual(stock["今日变化"]["Score变化"], 12)
        self.assertIn("MA5 高于 MA20", " ".join(stock["偏强证据"]))
        daily_markdown = create_evidence_report_content(evidence, "AI增强分析暂不可用。")
        stock_markdown = create_evidence_markdown_content(stock, evidence["市场环境"], "AI增强分析暂不可用。")
        email_body = daily_markdown.split("<!-- EMAIL_BODY_END -->", maxsplit=1)[0]
        self.assertIn("## 今天先看市场", email_body)
        self.assertIn("## 数据状态", email_body)
        self.assertIn("## 我的持仓：今天先核对什么", email_body)
        self.assertIn("已匹配本地收盘的市值 ¥1,234", email_body)
        self.assertIn("近 5 日 +3.00%", email_body)
        self.assertIn("成交量为近 5 日均量 1.30 倍", email_body)
        self.assertIn("1 只使用备用源", email_body)
        self.assertIn("## 今天最值得关注的公司", email_body)
        self.assertIn("结论：", email_body)
        self.assertNotIn("## 研究控制面板", email_body)
        self.assertNotIn("## 全量关注速览", email_body)
        self.assertNotIn("MACD", email_body)
        self.assertIn("## 市场环境", daily_markdown)
        self.assertIn("## 今日摘要", daily_markdown)
        self.assertIn("## 研究控制面板", daily_markdown)
        self.assertIn("## 行情来源与更新核对", daily_markdown)
        self.assertIn("东方财富 qfq kline / qfq", daily_markdown)
        self.assertIn("预测模型准入", daily_markdown)
        self.assertIn("## 今日重点与风险", daily_markdown)
        self.assertIn("## 20 日研究跟踪优先级", daily_markdown)
        self.assertIn("## 今日重点", daily_markdown)
        self.assertIn("## 重点股基本面与同业核对", daily_markdown)
        self.assertIn("测试主营业务", daily_markdown)
        self.assertIn("营业总收入同比增长 10.00%", daily_markdown)
        self.assertIn("静态PE 不适用（最新为非年报", daily_markdown)
        self.assertNotIn("静态PE None", daily_markdown)
        self.assertIn("## 全量关注速览", daily_markdown)
        self.assertIn("12.34", daily_markdown)
        self.assertIn("## 今日重点变化", daily_markdown)
        self.assertIn("## 重点关注股票深度解读", daily_markdown)
        self.assertIn("收盘价", daily_markdown)
        self.assertIn("观察评分 +12", daily_markdown)
        self.assertIn("近期价格状态", daily_markdown)
        self.assertIn("风险提示", daily_markdown)
        self.assertNotIn("出现变化，等待验证", daily_markdown)
        reader_content = daily_markdown.split("## 名词小抄", maxsplit=1)[0]
        self.assertIn("近 5 日平均价", reader_content)
        self.assertNotIn("MACD", reader_content)
        self.assertNotIn("RSI", reader_content)
        self.assertNotIn("短期动能", reader_content)
        self.assertIn("## 数据状态", stock_markdown)
        self.assertIn("## 偏强证据", stock_markdown)
        self.assertNotIn("未来 5 日上涨概率", daily_markdown + stock_markdown)
        self.assertNotIn("跑赢市场基准的实验概率", daily_markdown + stock_markdown)
        self.assertIn(
            "主要指数是否仍高于近 20 个交易日平均价格",
            create_rule_cross_signal_summary(evidence),
        )

    def test_missing_previous_day_changes_remain_insufficient(self):
        signal_file = self.output / "daily_signal_2026-07-24.json"
        data = json.loads(signal_file.read_text(encoding="utf-8"))
        data["stocks"][0]["数据状态"] = "缺少前一交易日快照，现有数据不足，无法判断。"
        data["stocks"][0]["今日变化"] = {
            "Score变化": "数据不足", "RSI变化": "数据不足",
            "MA5/MA20关系变化": "数据不足", "MACD状态变化": "数据不足",
        }
        write_json(signal_file, data)
        evidence = build_report_evidence(self.output, self.market, self.watchlist)
        self.assertEqual(evidence["关注股票"][0]["今日变化"]["Score变化"], "数据不足")

    def test_research_priority_records_top3_change_only_when_previous_snapshot_exists(self):
        previous = {
            "快照日期": "2026-07-23",
            "股票排行榜": [],
            "稳健研究候选": {"20日研究推荐": [
                {"股票代码": "000002", "股票名称": "昨日第一", "20日研究优先评分": 80},
                {"股票代码": "000001", "股票名称": "测试股", "20日研究优先评分": 60},
            ]},
        }
        current_file = self.output / "quant_snapshot_2026-07-24.json"
        current = json.loads(current_file.read_text(encoding="utf-8"))
        current["稳健研究候选"] = {"20日研究推荐": [
            {"股票代码": "000001", "股票名称": "测试股", "20日研究优先评分": 70, "推荐状态": "研究跟踪"},
            {"股票代码": "000003", "股票名称": "新进入", "20日研究优先评分": 65, "推荐状态": "研究跟踪"},
        ]}
        write_json(self.output / "quant_snapshot_2026-07-23.json", previous)
        write_json(current_file, current)

        evidence = build_report_evidence(self.output, self.market, self.watchlist)
        recommendations = evidence["优先研究标的"]
        self.assertIn("TOP3 内上升 1 位", recommendations[0]["TOP3动态"])
        self.assertEqual(recommendations[1]["TOP3动态"], "新进入 TOP3")


if __name__ == "__main__":
    unittest.main()
