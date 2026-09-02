import unittest
from datetime import date

import pandas as pd

from astock_core.data.market_data_sources import (
    BaoStockRawDailySource,
    MarketDataSourceError,
    RetryPolicy,
    TencentRawDailySource,
    fetch_daily_history,
    normalize_history,
)


def make_history():
    return pd.DataFrame(
        {
            "日期": ["2026-01-02", "2026-01-03"],
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 11.0],
            "最高": [10.7, 11.2],
            "最低": [9.8, 10.2],
            "成交量": [1000, 1200],
        }
    )


class FakeSource:
    adjustment = "qfq"

    def __init__(self, name, value=None, error=None):
        self.name = name
        self.value = value
        self.error = error
        self.calls = []

    def fetch(self, market_code):
        self.calls.append(market_code)
        if self.error:
            raise self.error
        return self.value.copy()


class FakeBaoStockResult:
    error_code = "0"
    error_msg = ""

    def __init__(self, rows):
        self.rows = list(rows)
        self.position = 0

    def next(self):
        if self.position >= len(self.rows):
            return False
        self.position += 1
        return True

    def get_row_data(self):
        return self.rows[self.position - 1]


class FakeBaoStock:
    def __init__(self):
        self.calls = []
        self.logged_out = False

    def login(self):
        self.calls.append(("login",))
        return FakeBaoStockResult([])

    def logout(self):
        self.logged_out = True

    def query_history_k_data_plus(self, *args, **kwargs):
        self.calls.append(("query", args, kwargs))
        return FakeBaoStockResult(
            [
                ["2026-01-02", "10.0", "10.7", "9.8", "10.5", "1000", "1"],
                ["2026-01-03", "10.5", "11.2", "10.2", "11.0", "1200", "1"],
            ]
        )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class MarketDataSourceTests(unittest.TestCase):
    def test_fallback_uses_one_complete_backup_source_and_keeps_provenance(self):
        primary = FakeSource("主源", error=ValueError("连接失败"))
        backup = FakeSource("备用源", value=make_history())

        result = fetch_daily_history("sz000001", sources=[primary, backup])

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.source, "备用源")
        self.assertEqual(result.attempted_sources, ("主源", "备用源"))
        self.assertEqual(result.failures, ({"数据源": "主源", "原因": "连接失败"},))
        self.assertEqual(len(result.history), 2)
        self.assertEqual(primary.calls, ["sz000001"])
        self.assertEqual(backup.calls, ["sz000001"])

    def test_first_source_success_does_not_call_backup(self):
        primary = FakeSource("主源", value=make_history())
        backup = FakeSource("备用源", value=make_history())

        result = fetch_daily_history("sh600000", sources=[primary, backup])

        self.assertFalse(result.used_fallback)
        self.assertEqual(backup.calls, [])

    def test_all_sources_fail_with_auditable_summary(self):
        with self.assertRaisesRegex(MarketDataSourceError, "主源.*备用源"):
            fetch_daily_history(
                "sz000001",
                sources=[FakeSource("主源", error=ValueError("超时")), FakeSource("备用源", error=ValueError("无数据"))],
            )

    def test_source_is_retried_before_falling_back(self):
        class FlakySource(FakeSource):
            def __init__(self):
                super().__init__("主源")
                self.remaining_failures = 1

            def fetch(self, market_code):
                self.calls.append(market_code)
                if self.remaining_failures:
                    self.remaining_failures -= 1
                    raise ValueError("临时超时")
                return make_history()

        primary = FlakySource()
        backup = FakeSource("备用源", value=make_history())
        result = fetch_daily_history(
            "sz000001",
            sources=[primary, backup],
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
        )

        self.assertEqual(primary.calls, ["sz000001", "sz000001"])
        self.assertEqual(backup.calls, [])
        self.assertEqual(result.request_attempts, 2)

    def test_failure_audit_records_retry_count(self):
        primary = FakeSource("主源", error=ValueError("临时超时"))
        backup = FakeSource("备用源", value=make_history())
        result = fetch_daily_history(
            "sz000001",
            sources=[primary, backup],
            retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
        )
        self.assertEqual(result.failures[0]["尝试次数"], 2)
        self.assertEqual(result.request_attempts, 3)

    def test_normalizer_rejects_invalid_ohlc_rows_but_keeps_valid_rows(self):
        history = make_history()
        history.loc[1, "最高"] = 9.0
        normalized = normalize_history(history)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.loc[0, "日期"].strftime("%Y-%m-%d"), "2026-01-02")

    def test_baostock_raw_source_requests_unadjusted_daily_history(self):
        api = FakeBaoStock()
        source = BaoStockRawDailySource(
            baostock_module=api,
            start_date="2020-01-01",
            end_date_factory=lambda: date(2026, 9, 1),
        )

        history = source.fetch("sh600000")

        self.assertEqual(len(history), 2)
        self.assertTrue(api.logged_out)
        _, args, kwargs = next(call for call in api.calls if call[0] == "query")
        self.assertEqual(args[0], "sh600000")
        self.assertEqual(kwargs["frequency"], "d")
        self.assertEqual(kwargs["adjustflag"], "3")
        self.assertEqual(kwargs["start_date"], "2020-01-01")
        self.assertEqual(kwargs["end_date"], "2026-09-01")

    def test_source_note_is_preserved_in_provenance(self):
        source = FakeSource("研究来源", value=make_history())
        source.source_note = "仅供研究的未复权日线。"

        provenance = fetch_daily_history("sh600000", sources=[source]).provenance("600000", "sh600000")

        self.assertEqual(provenance["来源说明"], "仅供研究的未复权日线。")

    def test_tencent_raw_source_uses_the_plain_kline_endpoint(self):
        calls = []

        def request_get(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(
                {"data": {"sh600000": {"day": [["2026-01-02", "10", "10.5", "10.7", "9.8", "1000"]]}}}
            )

        history = TencentRawDailySource(request_get=request_get).fetch("sh600000")

        self.assertEqual(len(history), 1)
        self.assertIn("/appstock/app/kline/kline", calls[0][0])
        self.assertEqual(calls[0][1]["params"]["param"], "sh600000,day,,,1200")


if __name__ == "__main__":
    unittest.main()
