"""模拟账本测试：不读取真实持仓，也不访问网络。"""

import tempfile
import unittest
from pathlib import Path

from astock_core.simulator.paper_portfolio import (
    create_snapshot_buy,
    load_simulator,
    save_simulator,
    summarize_simulator,
    upsert_simulator_cash,
)


class PaperPortfolioTests(unittest.TestCase):
    def candidate(self):
        return {"股票代码": "600839", "股票名称": "四川长虹", "现价": 10.0}

    def test_cash_and_confirmed_snapshot_buy_are_saved_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "simulator.json"
            simulator = upsert_simulator_cash(load_simulator(file_path), "测试模拟仓", 5000)
            simulator, transaction = create_snapshot_buy(
                simulator, "测试模拟仓", self.candidate(), 200, "strategy", "2026-09-02"
            )
            save_simulator(file_path, simulator)
            loaded = load_simulator(file_path)

        self.assertEqual(loaded["positions"][0]["quantity"], 200)
        self.assertEqual(transaction["手续费"], 5.0)
        self.assertEqual(loaded["cash"][0]["amount"], 2995.0)

    def test_buy_requires_lot_size_and_available_cash(self):
        simulator = upsert_simulator_cash(load_simulator(Path("missing.json")), "测试模拟仓", 1000)
        with self.assertRaisesRegex(ValueError, "100股"):
            create_snapshot_buy(simulator, "测试模拟仓", self.candidate(), 101, "strategy", "2026-09-02")
        with self.assertRaisesRegex(ValueError, "现金不足"):
            create_snapshot_buy(simulator, "测试模拟仓", self.candidate(), 200, "strategy", "2026-09-02")

    def test_summary_keeps_simulator_separate_from_quote_source(self):
        simulator = upsert_simulator_cash(load_simulator(Path("missing.json")), "测试模拟仓", 5000)
        simulator, _ = create_snapshot_buy(simulator, "测试模拟仓", self.candidate(), 100, "strategy", "2026-09-02")
        rows, summary = summarize_simulator(simulator, {"600839": {"close": 12.0, "date": "2026-09-03"}})

        self.assertEqual(rows[0]["模拟市值"], 1200.0)
        self.assertEqual(rows[0]["浮动盈亏"], 195.0)
        self.assertEqual(summary["模拟现金"], 3995.0)


if __name__ == "__main__":
    unittest.main()
