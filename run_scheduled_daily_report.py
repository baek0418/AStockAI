"""由 launchd 调用的 A 股交易日日报任务入口。"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
CALENDAR_FILE = PROJECT_DIRECTORY / "config" / "a_share_market_holidays.json"
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")


def get_logger():
    """返回仅写入项目 logs 目录的定时任务日志记录器。"""
    logger = logging.getLogger("scheduled_daily_report")
    if logger.handlers:
        return logger

    logs_directory = PROJECT_DIRECTORY / "logs"
    logs_directory.mkdir(exist_ok=True)
    log_file = logs_directory / f"scheduled_daily_report_{datetime.now(BEIJING_TIMEZONE):%Y-%m-%d}.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def load_market_holidays(calendar_file=CALENDAR_FILE):
    """加载版本化的交易所休市日；缺失年份时拒绝执行以避免误发邮件。"""
    try:
        calendar_data = json.loads(Path(calendar_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("无法读取 A 股交易日历配置。") from error

    holidays = calendar_data.get("holidays")
    if not isinstance(holidays, dict):
        raise ValueError("A 股交易日历配置格式错误。")

    parsed_holidays = {}
    for year, dates in holidays.items():
        if not isinstance(year, str) or not isinstance(dates, list):
            raise ValueError("A 股交易日历配置格式错误。")
        try:
            parsed_holidays[year] = {date.fromisoformat(value) for value in dates}
        except (TypeError, ValueError) as error:
            raise ValueError("A 股交易日历包含无效日期。") from error

    return parsed_holidays


def get_trading_day_status(target_date, holiday_calendar):
    """判断日期是否为已配置年度内的 A 股交易日。"""
    if target_date.weekday() >= 5:
        return False, "周末，A 股休市。"

    year_holidays = holiday_calendar.get(str(target_date.year))
    if year_holidays is None:
        return False, f"未配置 {target_date.year} 年 A 股交易日历，已安全跳过任务。"

    if target_date in year_holidays:
        return False, "交易所节假日休市。"

    return True, "A 股交易日。"


def run_scheduled_report(
    force=False,
    dry_run=False,
    now=None,
    runner=subprocess.run,
    project_directory=PROJECT_DIRECTORY,
    calendar_file=CALENDAR_FILE,
    logger=None,
):
    """在交易日执行既有流水线；不读取、打印或记录 .env 中的任何密钥。"""
    project_directory = Path(project_directory)
    logger = logger or get_logger()
    now = now or datetime.now(BEIJING_TIMEZONE)
    beijing_now = now.astimezone(BEIJING_TIMEZONE)
    today = beijing_now.date()

    if not (project_directory / ".env").is_file():
        logger.error("项目根目录缺少 .env，自动任务未执行。")
        return 1

    try:
        holiday_calendar = load_market_holidays(calendar_file)
    except ValueError as error:
        logger.error("%s 自动任务未执行。", error)
        return 1

    is_trading_day, reason = get_trading_day_status(today, holiday_calendar)
    if not force and not is_trading_day:
        logger.info("北京时间 %s：%s 不更新数据，也不发送邮件。", today, reason)
        return 0

    if dry_run:
        logger.info(
            "北京时间 %s：%s%s；试运行完成，未更新数据、未发送邮件。",
            today,
            reason,
            "已使用 --force，" if force else "",
        )
        return 0

    python_executable = project_directory / ".venv" / "bin" / "python"
    if not python_executable.is_file():
        logger.error("未找到 %s，自动任务未执行。", python_executable)
        return 1

    command = [str(python_executable), "pipeline.py", "--send-email"]
    logger.info("北京时间 %s：开始执行交易日日报任务。", today)
    try:
        result = runner(command, cwd=project_directory, check=False)
    except OSError:
        logger.exception("无法启动流水线；数据未更新，邮件未发送。")
        return 1

    if result.returncode != 0:
        logger.error("流水线执行失败（退出码 %s）；请查看 pipeline 日志中的数据更新或邮件错误。", result.returncode)
        return result.returncode or 1

    logger.info("交易日日报任务执行成功。")
    return 0


def main():
    """解析仅供手动运维使用的参数。"""
    parser = argparse.ArgumentParser(description="AStockAI launchd 交易日日报任务。")
    parser.add_argument("--force", action="store_true", help="忽略交易日判断，立即执行流水线。")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置与交易日，不运行流水线。")
    arguments = parser.parse_args()
    return run_scheduled_report(force=arguments.force, dry_run=arguments.dry_run)


if __name__ == "__main__":
    sys.exit(main())
