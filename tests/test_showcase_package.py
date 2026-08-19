"""脱敏展示包的内容边界测试。"""

import tempfile
import unittest
import zipfile
from pathlib import Path

from build_showcase_package import PACKAGE_NAME, build_package


class ShowcasePackageTests(unittest.TestCase):
    def test_package_contains_only_showcase_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [
                "market_data_adapter.py", "market_data_provenance.py", "qlib_local_export.py",
                "run_qlib_alpha_baseline.py", "prediction_features.py", "prediction_features_v2.py",
                "prediction_model.py", "prediction_data_audit.py",
            ]:
                (root / name).write_text("# safe source\n", encoding="utf-8")
            package_directory, archive = build_package(root)
            self.assertTrue((package_directory / "demo_app.py").is_file())
            self.assertTrue((package_directory / "sample_data" / "anonymous_price_history.csv").is_file())
            with zipfile.ZipFile(archive) as package:
                names = package.namelist()
            self.assertTrue(all(name.startswith(f"{PACKAGE_NAME}/") for name in names))
            forbidden = (".env", "models/", "output/", "logs/", "watchlist", "历史.csv")
            self.assertFalse(any(any(item in name for item in forbidden) for name in names))


if __name__ == "__main__":
    unittest.main()
