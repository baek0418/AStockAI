"""按需查询测试：全部使用临时目录和模拟行情，不访问真实接口。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from on_demand_analysis import (
    add_stock_to_watchlist,
    analyze_on_demand_stock,
    get_history_file,
    refresh_catalog,
    resolve_code_query,
    resolve_catalog_query,
)
from research_data import collect_stock_snapshots


def make_raw_history(days=35):
    """构造满足既有评分函数所需长度的模拟日线。"""
    raw_history = []
    for index in range(days):
        close = 10 + index * 0.1
        raw_history.append(
            [
                f"2026-06-{index + 1:02d}" if index < 30 else f"2026-07-{index - 29:02d}",
                str(close - 0.05),
                str(close),
                str(close + 0.1),
                str(close - 0.1),
                str(100000 + index * 1000),
            ]
        )
    return raw_history


class OnDemandAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_directory = self.root / "data" / "on_demand"
        self.output_directory = self.root / "output" / "on_demand"
        self.catalog_file = self.data_directory / "a_share_catalog.json"
        self.stock = {"code": "000001", "name": "测试银行"}

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_code_name_and_fuzzy_catalog_queries(self):
        catalog = [
            {"code": "600839", "name": "四川长虹"},
            {"code": "600000", "name": "浦发银行"},
            {"code": "000001", "name": "平安银行"},
        ]
        self.assertEqual(resolve_code_query("600839"), {"code": "600839", "name": "600839"})
        self.assertEqual(resolve_catalog_query("600839", catalog), [])
        self.assertEqual(resolve_catalog_query("四川长虹", catalog)[0]["code"], "600839")
        self.assertEqual(len(resolve_catalog_query("银行", catalog)), 2)
        self.assertEqual(resolve_catalog_query("900000", catalog), [])

    def test_code_query_and_tencent_analysis_do_not_depend_on_catalog_refresh(self):
        failed_refresh = refresh_catalog(
            request_get=lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ProxyError("代理失败")),
            catalog_file=self.catalog_file,
        )
        direct_stock = resolve_code_query("600839")
        result = analyze_on_demand_stock(
            direct_stock,
            get_history=lambda market_code: make_raw_history(),
            identity_fetcher=lambda code: {"code": code, "name": "四川长虹"},
            data_directory=self.data_directory,
            output_directory=self.output_directory,
            catalog_file=self.catalog_file,
        )

        self.assertEqual(failed_refresh["status"], "failed")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis"]["股票名称"], "四川长虹")
        self.assertIn("腾讯", result["analysis"]["行情来源"])
        self.assertIn("腾讯", result["analysis"]["名称目录来源"])

    def test_catalog_refresh_fetches_all_pages_before_saving(self):
        class FakeResponse:
            def __init__(self, data):
                self.data = data

            def raise_for_status(self):
                return None

            def json(self):
                return {"data": self.data}

        pages = {
            1: {"total": 3, "diff": [
                {"f12": "000001", "f14": "平安银行"},
                {"f12": "300750", "f14": "宁德时代"},
            ]},
            2: {"total": 3, "diff": [{"f12": "600589", "f14": "大位科技"}]},
        }

        def request_get(url, params, headers, timeout):
            self.assertIn("push2.eastmoney.com", url)
            self.assertEqual(headers["User-Agent"], "Mozilla/5.0")
            return FakeResponse(pages[params["pn"]])

        with patch("on_demand_analysis.CATALOG_PAGE_SIZE", 2):
            result = refresh_catalog(
                request_get=request_get,
                catalog_file=self.catalog_file,
                minimum_entries=3,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 3)
        self.assertEqual(resolve_catalog_query("大位科技", json.loads(
            self.catalog_file.read_text(encoding="utf-8")
        )["stocks"])[0]["code"], "600589")

    def test_incomplete_catalog_refresh_does_not_overwrite_existing_file(self):
        self.catalog_file.parent.mkdir(parents=True)
        self.catalog_file.write_text(json.dumps({"stocks": [self.stock]}), encoding="utf-8")

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"total": 180, "diff": [
                    {"f12": "600589", "f14": "大位科技"}
                ]}}

        result = refresh_catalog(
            request_get=lambda *args, **kwargs: FakeResponse(),
            catalog_file=self.catalog_file,
            minimum_entries=3000,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(json.loads(self.catalog_file.read_text(encoding="utf-8"))["stocks"], [self.stock])

    def test_on_demand_download_uses_only_isolated_directories(self):
        result = analyze_on_demand_stock(
            self.stock,
            get_history=lambda market_code: make_raw_history(),
            data_directory=self.data_directory,
            output_directory=self.output_directory,
            catalog_file=self.catalog_file,
        )

        self.assertEqual(result["status"], "success")
        analysis = result["analysis"]
        self.assertEqual(analysis["股票代码"], "000001")
        self.assertIn("日线", analysis["日线数据说明"])
        self.assertIn("今日变化", analysis["daily_signal"])
        self.assertTrue(get_history_file(self.stock, self.data_directory).is_file())
        self.assertTrue((self.output_directory / "on_demand_000001.json").is_file())

        cached_result = analyze_on_demand_stock(
            self.stock,
            get_history=lambda market_code: (_ for _ in ()).throw(AssertionError("不应重复下载")),
            data_directory=self.data_directory,
            output_directory=self.output_directory,
            catalog_file=self.catalog_file,
        )
        self.assertEqual(cached_result["status"], "cached")

    def test_download_failure_keeps_existing_cache_and_does_not_write_empty_csv(self):
        self.data_directory.mkdir(parents=True)
        history_file = get_history_file(self.stock, self.data_directory)
        history_file.write_text("保留的缓存", encoding="utf-8")

        result = analyze_on_demand_stock(
            self.stock,
            refresh=True,
            get_history=lambda market_code: (_ for _ in ()).throw(RuntimeError("网络失败")),
            data_directory=self.data_directory,
            output_directory=self.output_directory,
            catalog_file=self.catalog_file,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("已有本地缓存未被删除", result["message"])
        self.assertEqual(history_file.read_text(encoding="utf-8"), "保留的缓存")
        self.assertFalse((self.output_directory / "on_demand_000001.json").exists())

    def test_on_demand_subdirectory_is_not_scanned_by_research_data(self):
        data_directory = self.root / "data"
        on_demand_directory = data_directory / "on_demand"
        on_demand_directory.mkdir(parents=True)
        pd.DataFrame(
            [{"日期": "2026-07-24", "收盘": 10, "成交量": 1000}]
        ).to_csv(on_demand_directory / "000001_测试银行_历史.csv", index=False)

        self.assertEqual(collect_stock_snapshots(data_directory, {}), [])

    def test_stock_is_added_to_watchlist_only_after_explicit_call(self):
        watchlist_file = self.root / "watchlist.json"
        watchlist_file.write_text(json.dumps({"stocks": []}), encoding="utf-8")
        self.assertEqual(json.loads(watchlist_file.read_text(encoding="utf-8"))["stocks"], [])

        result = add_stock_to_watchlist(self.stock, watchlist_file)
        saved_watchlist = json.loads(watchlist_file.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(saved_watchlist["stocks"][0]["code"], "000001")
        self.assertEqual(add_stock_to_watchlist(self.stock, watchlist_file)["status"], "exists")


if __name__ == "__main__":
    unittest.main()
