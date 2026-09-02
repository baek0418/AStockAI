"""午后强势策略的纯函数测试：不访问公开行情。"""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from astock_core.strategies.afternoon_momentum import (
    daily_evidence,
    evaluate_candidate,
    initial_filter,
    minute_evidence,
    parse_quote_payload,
    run_afternoon_momentum_screen,
)


def quote_payload(change_pct="4.20"):
    fields = [""] * 54
    fields[1] = "测试股份"
    fields[2] = "000001"
    fields[3] = "10.00"
    fields[30] = "20260902150000"
    fields[32] = change_pct
    fields[33] = "10.10"
    fields[38] = "6.20"
    fields[44] = "80.00"
    fields[49] = "1.80"
    fields[51] = "9.90"
    return 'v_sz000001="' + "~".join(fields) + '";'


def daily_payload():
    rows = []
    for index in range(20):
        rows.append([f"2026-08-{index + 1:02d}", "1", str(index + 1), str(index + 1), "1", str(100 + index)])
    return {"data": {"sz000001": {"qfqday": rows}}}


def minute_payload(with_break=False):
    rows = [
        "1420 9.90 100 99000",
        "1430 10.10 200 202000",
        "1431 10.00 300 300000",
        "1440 10.00 400 400000",
    ]
    if with_break:
        rows[0] = "1420 9.80 100 99000"
    return {"data": {"sz000001": {"data": {"data": rows}}}}


class AfternoonMomentumTests(unittest.TestCase):
    def test_parser_and_initial_filter_keep_required_realtime_fields(self):
        quote = parse_quote_payload(quote_payload())["sz000001"]
        self.assertEqual(quote["name"], "测试股份")
        self.assertTrue(initial_filter(quote))

    def test_complete_evidence_passes_strict_conditions(self):
        quote = parse_quote_payload(quote_payload())["sz000001"]
        daily = daily_evidence(daily_payload(), "sz000001")
        minute = minute_evidence(minute_payload(), "sz000001", -1.0)
        result = evaluate_candidate(quote, daily, minute, -1.0)

        self.assertTrue(daily["volume_staircase"])
        self.assertTrue(daily["ma_bull"])
        self.assertTrue(minute["near_1430_new_high"])
        self.assertTrue(result["通过"])

    def test_intraday_break_excludes_candidate(self):
        quote = parse_quote_payload(quote_payload())["sz000001"]
        result = evaluate_candidate(
            quote,
            daily_evidence(daily_payload(), "sz000001"),
            minute_evidence(minute_payload(with_break=True), "sz000001", -1.0),
            -1.0,
        )

        self.assertFalse(result["通过"])
        self.assertIn("全天存在跌破", "；".join(result["未通过条件"]))

    def test_full_screen_uses_timestamped_quotes_and_returns_only_verified_candidate(self):
        class Response:
            def __init__(self, text=None, payload=None):
                self.content = (text or "").encode("gbk")
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        def request_get(url, **kwargs):
            if "qt.gtimg.cn" in url:
                return Response(text=quote_payload("-1.00") if url.endswith("sh000300") else quote_payload())
            if "fqkline" in url:
                return Response(payload=daily_payload())
            return Response(payload=minute_payload())

        catalog = [{"code": f"{index:06d}"} for index in range(1, 3001)]
        result = run_afternoon_momentum_screen(
            catalog,
            request_get=request_get,
            now=datetime(2026, 9, 2, 15, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["universe_count"], 3000)
        self.assertEqual([item["股票代码"] for item in result["candidates"]], ["000001"])


if __name__ == "__main__":
    unittest.main()
