"""在不含私有配置和行情的目录中验证首次使用页面，不访问网络。"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class OnboardingTests(unittest.TestCase):
    def test_clean_checkout_renders_without_data_or_env_and_never_auto_downloads(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(project / "app.py", root / "app.py")
            shutil.copy2(project / "watchlist.example.json", root / "watchlist.example.json")
            with patch("requests.sessions.Session.request", side_effect=AssertionError("首次打开不能联网")), \
                 patch("astock_core.runtime.background_tasks.start_background_task") as start_task:
                page = AppTest.from_file(str(root / "app.py")).run(timeout=30)
                self.assertEqual(list(page.exception), [])
                self.assertTrue(any(button.label == "初始化本地数据" for button in page.button))
                self.assertEqual((root / "watchlist.json").read_text(encoding="utf-8").strip(), '{\n  "stocks": []\n}')
                self.assertFalse((root / ".env").exists())
                start_task.assert_not_called()
                start_task.return_value = {"status": "failed", "message": "测试：服务暂不可用，可重试。"}
                next(button for button in page.button if button.label == "初始化本地数据").click().run()
                start_task.assert_called_once_with("first_setup", root.resolve())
                self.assertEqual(list(page.exception), [])
                self.assertTrue(any("可重试" in error.value for error in page.error))


if __name__ == "__main__":
    unittest.main()
