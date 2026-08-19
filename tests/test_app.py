"""本地 Web 查询窗口的数据辅助函数测试，不启动 Streamlit 或 AI。"""

import tempfile
import unittest
from pathlib import Path

from app import get_file_mtime_ns, get_signal_stock, get_watchlist_rows, load_json_file


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


if __name__ == "__main__":
    unittest.main()
