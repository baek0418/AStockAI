import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from astock_core.research.recommendation_evaluation import create_markdown, evaluate_recommendations


def write_history(path, closes):
    dates = pd.bdate_range("2025-01-01", periods=len(closes))
    pd.DataFrame({"日期": dates, "收盘": closes, "成交量": [1000] * len(closes)}).to_csv(path, index=False, encoding="utf-8-sig")


class RecommendationEvaluationTests(unittest.TestCase):
    def test_evaluation_is_rolling_and_compares_two_top3_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, market = root / "data", root / "data" / "market"
            market.mkdir(parents=True)
            closes = 100 + np.arange(130) * 0.2
            write_history(market / "沪深300_sh000300.csv", closes)
            # 使用真实股票名称，使评估仅加载当前股票池中的三只测试文件。
            for name, multiplier in (("贵州茅台", 1.0), ("宁德时代", 1.1), ("招商银行", 0.9)):
                write_history(data / f"{name}历史.csv", 20 + np.arange(130) * multiplier * 0.08)
            report = evaluate_recommendations(data, market)
            self.assertTrue(report["窗口"])
            self.assertIsNotNone(report["20日研究优先Top3汇总"])
            self.assertIsNotNone(report["综合评分Top3汇总"])
            self.assertIn("平均 20 日收益", create_markdown(report))


if __name__ == "__main__":
    unittest.main()
