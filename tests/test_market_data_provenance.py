"""本地行情复权来源核验测试。"""

import unittest

import pandas as pd

from market_data_provenance import classify_adjustment, compare_history


def history(close=10.0):
    return pd.DataFrame(
        {
            "日期": pd.bdate_range("2025-01-01", periods=301),
            "开盘": close,
            "收盘": close,
            "最高": close + 1,
            "最低": close - 1,
            "成交量": 1000.0,
        }
    )


class MarketDataProvenanceTests(unittest.TestCase):
    def test_compare_requires_all_ohlcv_fields_to_match(self):
        result = compare_history(history(), history(close=11.0))
        self.assertEqual(result["overlap_rows"], 301)
        self.assertEqual(result["match_rate"], 0.0)

    def test_classify_only_accepts_a_unique_full_match(self):
        result = classify_adjustment(history(), {"qfq": history(), "none": history(close=11.0)})
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["adjustment"], "qfq")

    def test_classify_rejects_ambiguous_or_insufficient_match(self):
        ambiguous = classify_adjustment(history(), {"qfq": history(), "none": history()})
        self.assertEqual(ambiguous["adjustment"], "unverified")
        short = classify_adjustment(history(), {"qfq": history().iloc[:10]})
        self.assertEqual(short["status"], "unmatched")

    def test_classify_accepts_a_single_latest_row_refresh_drift(self):
        local = history()
        refreshed = history()
        refreshed.loc[refreshed.index[-1], "收盘"] = 12.0
        result = classify_adjustment(local, {"qfq": refreshed})
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["adjustment"], "qfq")
        self.assertEqual(result["comparisons"]["qfq"]["mismatch_dates"], ["2026-02-25"])


if __name__ == "__main__":
    unittest.main()
