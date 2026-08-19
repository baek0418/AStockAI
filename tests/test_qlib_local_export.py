"""Qlib 本地导出层的选择与拆分测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qlib_local_export import load_verified_qfq_instruments, write_qlib_source_files


class QlibLocalExportTests(unittest.TestCase):
    def test_only_verified_qfq_is_selected_and_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "audit.json"
            audit_path.write_text(
                json.dumps(
                    {"files": [
                        {"market_code": "sh600000", "status": "verified", "adjustment": "qfq"},
                        {"market_code": "sz000001", "status": "verified", "adjustment": "none"},
                        {"market_code": "sz000002", "status": "unmatched", "adjustment": "unverified"},
                    ]}
                ),
                encoding="utf-8",
            )
            instruments = load_verified_qfq_instruments(audit_path)
            dataset = pd.DataFrame(
                {
                    "instrument": ["sh600000", "sz000001"],
                    "date": ["2025-01-01", "2025-01-01"],
                    "open": [1.0, 2.0], "high": [1.1, 2.1], "low": [0.9, 1.9],
                    "close": [1.0, 2.0], "volume": [100.0, 200.0],
                }
            )
            files, skipped = write_qlib_source_files(dataset, instruments, root / "source")
            self.assertEqual([file.name for file in files], ["sh600000.csv"])
            self.assertEqual(skipped, ["sz000001"])
            self.assertEqual(pd.read_csv(files[0]).columns.tolist(), ["date", "open", "high", "low", "close", "volume"])


if __name__ == "__main__":
    unittest.main()
