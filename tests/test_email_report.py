"""email_report 的模拟 SMTP 测试，不会建立真实网络连接。"""

import smtplib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from email_report import extract_email_content, render_report_html, send_daily_report


class FakeSMTP:
    instances = []

    def __init__(self, host, port, context, timeout):
        self.host = host
        self.port = port
        self.context = context
        self.timeout = timeout
        self.login_args = None
        self.message = None
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def login(self, username, password):
        self.login_args = (username, password)

    def send_message(self, message):
        self.message = message


class AuthenticationErrorSMTP(FakeSMTP):
    def login(self, username, password):
        raise smtplib.SMTPAuthenticationError(535, b"Authentication failed")


class RecipientErrorSMTP(FakeSMTP):
    def send_message(self, message):
        raise smtplib.SMTPRecipientsRefused({"invalid@example.com": (550, b"Rejected")})


class ConnectionErrorSMTP:
    def __init__(self, *args, **kwargs):
        raise OSError("network unavailable")


class EmailReportTests(unittest.TestCase):
    def setUp(self):
        self.environment_patch = patch.dict("os.environ", {}, clear=True)
        self.environment_patch.start()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.report_file = self.directory / "每日关注股票日报_2026-07-24.md"
        self.report_content = "# AStockAI 每日关注股票日报\n\n测试内容。\n"
        self.report_file.write_text(self.report_content, encoding="utf-8")
        self.env_file = self.directory / ".env"
        self.env_file.write_text(
            "\n".join(
                [
                    "SMTP_HOST=smtp.qq.com",
                    "SMTP_PORT=465",
                    "SMTP_USERNAME=sender@qq.com",
                    "SMTP_PASSWORD=test-authorisation-code",
                    "SMTP_FROM=sender@qq.com",
                    "REPORT_RECIPIENT=recipient@example.com",
                ]
            ),
            encoding="utf-8",
        )
        FakeSMTP.instances = []

    def tearDown(self):
        self.temporary_directory.cleanup()
        self.environment_patch.stop()

    def test_builds_html_report_with_text_fallback_and_attachment(self):
        result = send_daily_report(self.report_file, self.env_file, smtp_class=FakeSMTP)

        self.assertEqual(result["status"], "success")
        smtp = FakeSMTP.instances[0]
        self.assertEqual(smtp.host, "smtp.qq.com")
        self.assertEqual(smtp.port, 465)
        self.assertEqual(smtp.login_args[0], "sender@qq.com")
        self.assertEqual(
            smtp.message.get_body(preferencelist=("plain",)).get_content(), self.report_content
        )
        html_body = smtp.message.get_body(preferencelist=("html",)).get_content()
        self.assertIn("<h1", html_body)
        self.assertIn("AStockAI", html_body)
        self.assertIn("每日研究报告", html_body)
        attachment = list(smtp.message.iter_attachments())[0]
        self.assertEqual(attachment.get_filename(), self.report_file.name)
        self.assertEqual(attachment.get_content(), self.report_content)

    def test_html_renderer_escapes_report_content_and_formats_structure(self):
        rendered = render_report_html(
            "# 标题\n\n## 市场环境\n\n- **重点** <script>alert(1)</script>\n\n正文"
        )

        self.assertIn("<h1", rendered)
        self.assertIn("<h2", rendered)
        self.assertIn("<strong>重点</strong>", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)

    def test_email_content_excludes_markdown_appendix(self):
        content = "# 日报\n\n核心结论\n\n<!-- EMAIL_BODY_END -->\n\n<details>附录</details>"
        self.assertEqual(extract_email_content(content), "# 日报\n\n核心结论\n")

    def test_returns_safe_chinese_message_for_authentication_error(self):
        result = send_daily_report(
            self.report_file, self.env_file, smtp_class=AuthenticationErrorSMTP
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("授权码", result["message"])
        self.assertNotIn("test-authorisation-code", result["message"])

    def test_returns_chinese_message_for_recipient_rejection(self):
        result = send_daily_report(
            self.report_file, self.env_file, smtp_class=RecipientErrorSMTP
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("收件人", result["message"])

    def test_returns_chinese_message_for_connection_error(self):
        result = send_daily_report(
            self.report_file, self.env_file, smtp_class=ConnectionErrorSMTP
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("网络或 SMTP 连接异常", result["message"])


if __name__ == "__main__":
    unittest.main()
