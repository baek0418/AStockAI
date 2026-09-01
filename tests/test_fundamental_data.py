import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import requests

from fundamental_data import (
    build_valuation_observation,
    build_industry_peer_comparison,
    collect_fundamental_snapshot,
    get_snapshot_file,
    refresh_watchlist_fundamentals,
    summarize_fundamental_evidence,
)


def sample_records():
    return [{
        "SECUCODE": "000001.SZ", "SECURITY_NAME_ABBR": "测试银行",
        "REPORT_DATE": "2026-06-30", "REPORT_TYPE": "中报", "NOTICE_DATE": "2026-08-28",
        "CURRENCY": "CNY", "TOTALOPERATEREVE": 1000000000, "PARENTNETPROFIT": 100000000,
        "TOTALOPERATEREVETZ": 12.5, "PARENTNETPROFITTZ": 8.0, "ROEJQ": 9.1,
        "ZCFZL": 60.0, "MGJYXJJE": 0.8,
        "EPSJB": 1.0, "BPS": 10.0,
    }]


class FundamentalDataTests(unittest.TestCase):
    def test_collect_writes_traceable_snapshot_only_after_valid_response(self):
        class Response:
            def raise_for_status(self):
                return None

            def __init__(self, records):
                self.records = records

            def json(self):
                return {"result": {"data": self.records}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            profile = [{"EM2016": "金融-银行", "INDUSTRYCSRC1": "货币金融服务", "MAIN_BUSINESS": "商业银行业务"}]
            result = collect_fundamental_snapshot(
                "000001", directory,
                request_get=lambda url, params, headers, timeout: Response(
                    profile if "v1/get" in url else sample_records()
                ),
            )
            self.assertEqual(result["status"], "success")
            saved = json.loads(get_snapshot_file("000001", directory).read_text(encoding="utf-8"))
            self.assertEqual(saved["最新报告"]["报告期"], "2026-06-30")
            self.assertEqual(saved["公司与行业画像"]["字段"]["所属行业"], "金融-银行")
            self.assertIn("cninfo.com.cn", saved["官方核验页"])
            evidence = summarize_fundamental_evidence(saved)
            self.assertEqual(evidence["数据状态"], "可用")
            self.assertTrue(any("归母净利润" in item for item in evidence["事实"]))
            self.assertTrue(any("主营业务" in item for item in evidence["事实"]))

    def test_failure_keeps_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            target = get_snapshot_file("000001", directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"keep": true}', encoding="utf-8")
            result = collect_fundamental_snapshot(
                "000001", directory,
                request_get=lambda *args, **kwargs: (_ for _ in ()).throw(requests.RequestException("网络失败")),
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"keep": True})

    def test_valuation_uses_latest_local_close_and_only_annual_eps_for_static_pe(self):
        annual = {
            "数据状态": "可用", "价格日期": "2026-08-27", "报告期": "2025-12-31",
            "指标": {"每股净资产": 10.0, "每股收益(基本)": 1.0},
        }
        annual_valuation = build_valuation_observation(annual, 12.5)
        self.assertEqual(annual_valuation["市净率(PB)"], 1.25)
        self.assertEqual(annual_valuation["静态市盈率(PE)"], 12.5)
        interim = dict(annual, 报告期="2026-06-30")
        self.assertIsNone(build_valuation_observation(interim, 12.5)["静态市盈率(PE)"])

    def test_peer_comparison_requires_same_industry_and_report_period(self):
        def snapshot(code, revenue_yoy, profit_yoy, roe, debt):
            return {
                "数据状态": "可用", "股票代码": code,
                "最新报告": {"报告期": "2026-06-30", "指标": {
                    "营业总收入同比增长": revenue_yoy, "归母净利润同比增长": profit_yoy,
                    "净资产收益率(加权)": roe, "资产负债率": debt,
                }},
                "公司与行业画像": {"数据状态": "可用", "字段": {"所属行业": "测试行业"}},
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            target = snapshot("000001", 15, 10, 8, 40)
            for item in (target, snapshot("000002", 10, 8, 6, 30), snapshot("000003", 20, 12, 10, 60)):
                get_snapshot_file(item["股票代码"], directory).write_text(
                    json.dumps(item, ensure_ascii=False), encoding="utf-8"
                )
            comparison = build_industry_peer_comparison(target, directory)
            self.assertEqual(comparison["数据状态"], "可用")
            self.assertEqual(comparison["指标比较"]["营业总收入同比增长"]["同业排名"], 2)
            self.assertEqual(comparison["指标比较"]["资产负债率"]["同业排名"], 2)

    def test_watchlist_refresh_skips_fresh_snapshot_and_updates_only_stale_one(self):
        class Response:
            def raise_for_status(self):
                return None

            def __init__(self, records):
                self.records = records

            def json(self):
                return {"result": {"data": self.records}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "fundamentals"
            watchlist = root / "watchlist.json"
            watchlist.write_text(json.dumps({"stocks": [
                {"code": "000001", "enable": True}, {"code": "000002", "enable": True},
            ]}), encoding="utf-8")
            fresh = {
                "数据状态": "可用", "下载时间": "2026-08-26 10:00:00",
                "股票代码": "000001", "最新报告": {}, "公司与行业画像": {},
            }
            directory.mkdir()
            get_snapshot_file("000001", directory).write_text(json.dumps(fresh), encoding="utf-8")
            calls = []

            def request_get(url, params, headers, timeout):
                calls.append(params)
                if "reportName" in params:
                    records = [{"EM2016": "测试行业"}]
                else:
                    records = sample_records()
                    records[0]["SECUCODE"] = params["filter"].split('"')[1]
                return Response(records)

            result = refresh_watchlist_fundamentals(
                watchlist, directory, max_age_days=7, request_get=request_get,
                now=datetime(2026, 8, 27, 10, 0, 0),
            )
            self.assertEqual(result["已更新"], ["000002"])
            self.assertEqual(result["仍有效无需更新"], ["000001"])
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
