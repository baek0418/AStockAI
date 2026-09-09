import json
import tempfile
import unittest
from unittest.mock import Mock
from datetime import datetime
from pathlib import Path

from astock_core.research.research_universe import (
    EXPECTED_CSI300_SIZE,
    fetch_csi300_constituents,
    load_research_universe,
    save_csi300_snapshot,
)


class ResearchUniverseTests(unittest.TestCase):
    def test_disabled_refresh_does_not_request_supplier(self):
        getter = Mock()
        with self.assertRaisesRegex(ValueError, "禁用东方财富"):
            fetch_csi300_constituents(getter)
        getter.assert_not_called()

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

if __name__ == "__main__":
    unittest.main()
