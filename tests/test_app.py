"""本地 Web 查询窗口的数据辅助函数测试，不启动 Streamlit 或 AI。"""

import tempfile
import unittest
from pathlib import Path

from app import (
    build_fundamental_display_data,
    get_file_mtime_ns,
    get_signal_stock,
    get_watchlist_rows,
    load_json_file,
)


class AppDataHelpersTests(unittest.TestCase):
    def test_file_mtime_and_json_loading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_file = Path(temporary_directory) / "sample.json"
            json_file.write_text('{"value": 1}', encoding="utf-8")

            self.assertIsInstance(get_file_mtime_ns(json_file), int)
            self.assertEqual(load_json_file(json_file, "测试文件"), {"value": 1})
            self.assertIsNone(get_file_mtime_ns(Path(temporary_directory) / "missing.json"))

    def test_signal_matches_code_before_name_and_watchlist_is_read_only_data(self):
        signal = {
            "stocks": [
                {"股票代码": "600839", "股票名称": "四川长虹", "今日变化": {}},
                {"股票代码": "000001", "股票名称": "平安银行", "今日变化": {}},
            ]
        }
        stock_record = {"code": "600839", "name": "长虹"}
        self.assertEqual(get_signal_stock(signal, stock_record)["股票名称"], "四川长虹")

        rows = get_watchlist_rows(
            {"stocks": [{"code": "600839", "name": "四川长虹", "tags": ["AI"], "enable": True}]}
        )
        self.assertEqual(rows, [{"代码": "600839", "名称": "四川长虹", "别名": "", "优先级": "", "启用": True, "标签": "AI", "备注": ""}])

    def test_fundamental_display_keeps_only_saved_facts_and_available_peer_metrics(self):
        display = build_fundamental_display_data(
            {
                "基本面研究证据": {
                    "数据状态": "可用",
                    "报告期": "2026-06-30",
                    "公告日期": "2026-08-20",
                    "来源": "测试来源",
                    "官方核验页": "https://example.com/report",
                    "指标": {"营业总收入": 100, "每股净资产": 12.5},
                    "事实": ["所属行业：银行（按来源原始描述）。", "主观结论：不应展示。"],
                },
                "估值观察": {"数据状态": "可用", "市净率(PB)": 1.2},
                "行业同业比较": {"数据状态": "数据不足：本地快照不足。"},
            }
        )
        self.assertEqual(display["报告期"], "2026-06-30")
        self.assertEqual(display["财务指标"], [
            {"指标": "营业总收入", "数值": 100, "口径": "来源原始口径"},
            {"指标": "每股净资产", "数值": 12.5, "口径": "来源原始口径"},
        ])
        self.assertEqual(display["公司与行业事实"], ["所属行业：银行（按来源原始描述）。"])
        self.assertEqual(display["同业比较"]["数据状态"], "数据不足：本地快照不足。")

    def test_fundamental_display_preserves_explicit_missing_snapshot(self):
        display = build_fundamental_display_data(
            {"基本面研究证据": {"数据状态": "数据不足：未下载基本面快照。"}}
        )
        self.assertEqual(display["数据状态"], "数据不足：未下载基本面快照。")
        self.assertEqual(display["财务指标"], [])


if __name__ == "__main__":
    unittest.main()
