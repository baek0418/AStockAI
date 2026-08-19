import unittest

from network_healthcheck import probe


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


class NetworkHealthcheckTests(unittest.TestCase):
    def test_expected_status_is_reachable(self):
        result = probe("测试", "https://example.test", {200}, request_get=lambda *args, **kwargs: Response(200))
        self.assertEqual(result["状态"], "可达")

    def test_request_error_is_safe(self):
        def fail(*args, **kwargs):
            raise __import__("requests").ConnectionError("not connected")
        result = probe("测试", "https://example.test", {200}, request_get=fail)
        self.assertEqual(result["状态"], "不可达")
        self.assertNotIn("not connected", result["说明"])


if __name__ == "__main__":
    unittest.main()
