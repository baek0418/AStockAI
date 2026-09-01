import unittest

import pandas as pd

from market_data_sources import MarketDataSourceError, RetryPolicy, fetch_daily_history, normalize_history


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


if __name__ == "__main__":
    unittest.main()
