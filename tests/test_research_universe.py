import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from research_universe import (
    EXPECTED_CSI300_SIZE,
    fetch_csi300_constituents,
    load_research_universe,
    save_csi300_snapshot,
)


class ResearchUniverseTests(unittest.TestCase):
    def test_fetches_all_constituent_pages_before_validating_snapshot(self):
        rows = [
            {"f12": f"60{number:04d}", "f14": f"测试{number}"}
            for number in range(EXPECTED_CSI300_SIZE)
        ]
        requested_pages = []

        class Response:
            def __init__(self, page_rows):
                self.page_rows = page_rows

            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"total": EXPECTED_CSI300_SIZE, "diff": self.page_rows}}

        def fake_get(_url, *, params, **_kwargs):
            requested_pages.append(params["pn"])
            start = (params["pn"] - 1) * params["pz"]
            return Response(rows[start:start + params["pz"]])

        stocks = fetch_csi300_constituents(fake_get)
        self.assertEqual(requested_pages, [1, 2, 3])
        self.assertEqual(len(stocks), EXPECTED_CSI300_SIZE)
        self.assertEqual(stocks[0]["code"], "600000")

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
