"""通过 QQ 邮箱 SMTP SSL 发送带 HTML 正文的 AStockAI 每日日报。"""

import html
import os
import re
import smtplib
import socket
import ssl
from email.message import EmailMessage
from pathlib import Path


REQUIRED_SMTP_SETTINGS = (
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "SMTP_FROM",
    "REPORT_RECIPIENT",
)
EMAIL_BODY_END_MARKER = "<!-- EMAIL_BODY_END -->"


class EmailReportError(Exception):
    """表示可安全展示给用户的邮件发送错误。"""


def extract_email_content(report_content):
    """邮件仅发送日报正文；完整 Markdown 仍作为附件保留。"""
    return report_content.split(EMAIL_BODY_END_MARKER, maxsplit=1)[0].rstrip() + "\n"


def render_report_html(report_content):
    """将受控 Markdown 日报渲染为兼容主流邮箱的简洁 HTML 正文。

    原始 Markdown 仍作为附件保留；这里仅支持日报实际使用的标题、段落、
    无序列表和加粗，避免引入额外依赖或把报告内容当作 HTML 执行。
    """
    blocks = []
    list_items = []
    paragraph_lines = []

    def format_inline(text):
        escaped = html.escape(text.strip())
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

    def flush_list():
        if list_items:
            blocks.append(
                '<ul style="margin:8px 0 18px;padding-left:22px;color:#334155;line-height:1.7">'
                + "".join(f"<li>{item}</li>" for item in list_items)
                + "</ul>"
            )
            list_items.clear()

    def flush_paragraph():
        if paragraph_lines:
            text = " ".join(paragraph_lines)
            blocks.append(
                '<p style="margin:8px 0 14px;color:#334155;line-height:1.75">'
                + format_inline(text)
                + "</p>"
            )
            paragraph_lines.clear()

    for raw_line in report_content.splitlines():
        line = raw_line.strip()
        if not line or set(line) == {"-"}:
            flush_list()
            flush_paragraph()
            continue

        heading = re.fullmatch(r"(#{1,3})\s+(.+)", line)
        if heading:
            flush_list()
            flush_paragraph()
            level = len(heading.group(1))
            title = format_inline(heading.group(2))
            if level == 1:
                blocks.append(
                    '<h1 style="margin:0 0 8px;font-size:26px;line-height:1.3;color:#0f172a">'
                    + title
                    + "</h1>"
                )
            elif level == 2:
                blocks.append(
                    '<h2 style="margin:30px 0 12px;padding:0 0 8px;border-bottom:1px solid #dbe4ee;'
                    'font-size:19px;color:#0f3d66">'
                    + title
                    + "</h2>"
                )
            else:
                blocks.append(
                    '<h3 style="margin:22px 0 8px;font-size:16px;color:#0f172a">'
                    + title
                    + "</h3>"
                )
            continue

        bullet = re.fullmatch(r"[-*]\s+(.+)", line)
        if bullet:
            flush_paragraph()
            list_items.append(format_inline(bullet.group(1)))
            continue

        flush_list()
        paragraph_lines.append(line)

    flush_list()
    flush_paragraph()

    return """<!doctype html>
<html lang="zh-CN">
  <body style="margin:0;padding:0;background:#f3f6fa;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',Arial,sans-serif">
    <main style="max-width:720px;margin:0 auto;padding:24px 12px">
      <section style="background:#ffffff;border:1px solid #dbe4ee;border-radius:12px;padding:28px;box-shadow:0 2px 8px rgba(15,23,42,.04)">
        <div style="margin:0 0 20px;color:#64748b;font-size:13px">AStockAI · 每日研究报告</div>
        """ + "\n".join(blocks) + """
        <footer style="margin-top:32px;padding-top:16px;border-top:1px solid #dbe4ee;color:#64748b;font-size:12px;line-height:1.6">
          本邮件为量化研究信息展示，不构成投资建议。原始 Markdown 日报已随邮件附上。
        </footer>
      </section>
    </main>
  </body>
</html>"""


def read_dotenv(env_file):
    """读取简单的 .env 文件，且不输出其中任何配置值。"""
    settings = {}
    env_path = Path(env_file)

    if not env_path.is_file():
        return settings

    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                value = value[1:-1]
            settings[key] = value

    return settings


def load_smtp_settings(env_file=None):
    """读取 .env 及进程环境变量中的 SMTP 配置并完成基础校验。"""
    project_directory = Path(__file__).parents[2]
    file_settings = read_dotenv(env_file or project_directory / ".env")
    settings = {
        key: os.environ.get(key, file_settings.get(key, "")).strip()
        for key in REQUIRED_SMTP_SETTINGS
    }
    missing_settings = [key for key, value in settings.items() if not value]
    if missing_settings:
        raise EmailReportError(
            "邮件配置不完整，缺少：" + "、".join(missing_settings) + "。请检查 .env 文件。"
        )

    try:
        settings["SMTP_PORT"] = int(settings["SMTP_PORT"])
    except ValueError as error:
        raise EmailReportError("SMTP_PORT 必须是有效的端口号。") from error

    if not 1 <= settings["SMTP_PORT"] <= 65535:
        raise EmailReportError("SMTP_PORT 必须在 1 到 65535 之间。")

    settings["REPORT_RECIPIENT"] = [
        recipient.strip()
        for recipient in settings["REPORT_RECIPIENT"].replace(";", ",").split(",")
        if recipient.strip()
    ]
    if not settings["REPORT_RECIPIENT"]:
        raise EmailReportError("REPORT_RECIPIENT 中没有有效收件人。")

    return settings


def build_report_message(report_file, sender, recipients):
    """构造 HTML 内容报告、纯文本兜底及原始 Markdown 附件。"""
    report_path = Path(report_file)
    try:
        report_content = report_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EmailReportError("无法读取本地日报文件，邮件未发送。") from error

    message = EmailMessage()
    message["Subject"] = f"AStockAI {report_path.stem}"
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    email_content = extract_email_content(report_content)
    # 为不显示 HTML 的客户端提供可读的短日报兜底；完整内容见 Markdown 附件。
    message.set_content(email_content, subtype="plain", charset="utf-8")
    message.add_alternative(render_report_html(email_content), subtype="html", charset="utf-8")
    message.add_attachment(
        report_content,
        subtype="markdown",
        filename=report_path.name,
    )
    return message


def send_daily_report(report_file, env_file=None, smtp_class=smtplib.SMTP_SSL):
    """发送日报并返回不含敏感信息的结构化结果。"""
    report_path = Path(report_file)
    if report_path.suffix.lower() != ".md" or not report_path.is_file():
        return {"status": "failed", "message": "日报文件不存在或不是 Markdown 文件，邮件未发送。"}

    try:
        settings = load_smtp_settings(env_file)
        message = build_report_message(
            report_path, settings["SMTP_FROM"], settings["REPORT_RECIPIENT"]
        )
        context = ssl.create_default_context()
        with smtp_class(
            settings["SMTP_HOST"],
            settings["SMTP_PORT"],
            context=context,
            timeout=20,
        ) as smtp:
            smtp.login(settings["SMTP_USERNAME"], settings["SMTP_PASSWORD"])
            smtp.send_message(message)
    except EmailReportError as error:
        return {"status": "failed", "message": str(error)}
    except smtplib.SMTPAuthenticationError:
        return {
            "status": "failed",
            "message": "QQ 邮箱账号或授权码验证失败，请检查 SMTP_USERNAME 和 SMTP_PASSWORD。",
        }
    except smtplib.SMTPRecipientsRefused:
        return {
            "status": "failed",
            "message": "收件人被 SMTP 服务器拒绝，请检查 REPORT_RECIPIENT。",
        }
    except (
        smtplib.SMTPConnectError,
        smtplib.SMTPServerDisconnected,
        socket.gaierror,
        socket.timeout,
        TimeoutError,
        ConnectionError,
        OSError,
    ):
        return {
            "status": "failed",
            "message": "网络或 SMTP 连接异常，邮件未发送；本地日报文件已保留。",
        }
    except smtplib.SMTPException:
        return {"status": "failed", "message": "SMTP 邮件发送失败；本地日报文件已保留。"}

    return {
        "status": "success",
        "message": "日报邮件发送成功。",
        "report_file": str(report_path),
    }
