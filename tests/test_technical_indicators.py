import unittest

import numpy as np
import pandas as pd

from technical_indicators import TECHNICAL_INDICATOR_COLUMNS, calculate_technical_indicators, latest_technical_indicators


def make_history(days=80):
    close = 10 + np.arange(days) * 0.1 + np.sin(np.arange(days) / 4)
    return pd.DataFrame(
        {
            "日期": pd.bdate_range("2026-01-01", periods=days),
            "开盘": close - 0.1,
            "收盘": close,
            "最高": close + 0.2,
            "最低": close - 0.3,
            "成交量": 1000 + np.arange(days) * 10,
        }
    )


class TechnicalIndicatorTests(unittest.TestCase):
    def test_indicator_frame_has_all_declared_columns(self):
        indicators = calculate_technical_indicators(make_history())
        self.assertEqual(indicators.columns.tolist(), ["日期", *TECHNICAL_INDICATOR_COLUMNS])
        self.assertTrue(indicators["rsi_14"].iloc[-1] >= 0)
        self.assertTrue(indicators["rsi_14"].iloc[-1] <= 100)
        self.assertTrue(pd.notna(indicators["boll_position"].iloc[-1]))

    def test_future_rows_do_not_change_previous_indicators(self):
        original = calculate_technical_indicators(make_history(60))
        extended = calculate_technical_indicators(make_history(75))
        pd.testing.assert_frame_equal(
            original.iloc[:55].reset_index(drop=True),
            extended.iloc[:55].reset_index(drop=True),
        )

    def test_latest_indicators_are_explicit_about_their_date(self):
        history = make_history()
        latest = latest_technical_indicators(history)
        self.assertEqual(latest["指标日期"], history["日期"].iloc[-1].strftime("%Y-%m-%d"))
        self.assertIn("macd_dif", latest)
        self.assertIn("momentum_20d", latest)


if __name__ == "__main__":
    unittest.main()
