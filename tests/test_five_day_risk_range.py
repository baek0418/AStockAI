import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from astock_core.analysis.five_day_risk_range import build_five_day_risk_range


class FiveDayRiskRangeTests(unittest.TestCase):
    def test_builds_a_local_risk_range_only_with_validated_report_and_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_directory = root / "raw"
            research_directory = root / "research"
            raw_directory.mkdir()
            research_directory.mkdir()
            pd.DataFrame(
                {
                    "日期": pd.bdate_range("2026-01-01", periods=25),
                    "收盘": [100 + index * 0.4 + (index % 3) for index in range(25)],
                }
            ).to_csv(raw_directory / "测试股历史.csv", index=False, encoding="utf-8-sig")
            (research_directory / "return_interval_5d_raw_selection_2026-09-01.json").write_text(
                json.dumps(
                    {
                        "数据可得性审计": {"严格时点价格依据可用": True},
                        "推荐区间方法": {
                            "方法": "历史波动率风险范围",
                            "指标": {"覆盖率": 0.8, "样本数": 1000},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = build_five_day_risk_range(
                {"code": "600000", "name": "测试股"}, raw_directory, research_directory
            )

        self.assertEqual(result["状态"], "可用")
        self.assertLess(result["下限价格"], result["本地收盘"])
        self.assertGreater(result["上限价格"], result["本地收盘"])
        self.assertEqual(result["历史覆盖率"], 80.0)

    def test_refuses_to_infer_when_raw_history_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "research").mkdir()
            result = build_five_day_risk_range({"code": "600000", "name": "测试股"}, root / "raw", root / "research")
        self.assertEqual(result["状态"], "数据不足")


if __name__ == "__main__":
    unittest.main()
