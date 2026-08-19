import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from import_csi300_snapshot import import_snapshot
from research_universe import EXPECTED_CSI300_SIZE, load_research_universe, save_csi300_snapshot


class ResearchUniverseTests(unittest.TestCase):
    def test_snapshot_round_trip_requires_exactly_300_unique_stocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "research_universe.json"
            stocks = [
                {"code": f"60{number:04d}", "name": f"测试{number}", "market_code": f"sh60{number:04d}"}
                for number in range(EXPECTED_CSI300_SIZE)
            ]
            snapshot = save_csi300_snapshot(stocks, root / "universe", datetime(2026, 7, 28))
            config.write_text(json.dumps({"enabled": True, "snapshot_file": str(snapshot.relative_to(root))}), encoding="utf-8")
            loaded = load_research_universe(config)
            self.assertEqual(len(loaded), EXPECTED_CSI300_SIZE)
            self.assertEqual(loaded[0]["source"], "research:csi300")

    def test_csv_import_requires_complete_unique_constituents_before_enabling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "csi300.csv"
            source.write_text(
                "股票代码,股票名称\n" + "\n".join(
                    f"60{number:04d},测试{number}" for number in range(EXPECTED_CSI300_SIZE)
                ),
                encoding="utf-8",
            )
            snapshot = import_snapshot(source, "2026-07-28", root / "research_universe.json")
            self.assertTrue(snapshot.is_file())
            self.assertEqual(len(load_research_universe(root / "research_universe.json")), EXPECTED_CSI300_SIZE)


if __name__ == "__main__":
    unittest.main()
