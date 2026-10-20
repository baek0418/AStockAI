"""正式日线只接收已结束交易日的数据，按本地休市表核验日期。"""
import json
from datetime import datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def latest_completed_session(now=None):
    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    if current.tzinfo is not None:
        current = current.astimezone(ZoneInfo("Asia/Shanghai"))
    holidays = json.loads((Path(__file__).parents[2] / "config/a_share_market_holidays.json").read_text(encoding="utf-8"))["holidays"]
    day = current.date()
    # 为收盘后的公开源更新留出时间。
    if current.time() < time(15, 30):
        day -= timedelta(days=1)
    while True:
        if str(day.year) not in holidays:
            raise ValueError(f"缺少 {day.year} 年交易日历，不能确认日线新鲜度。")
        if day.weekday() < 5 and day.isoformat() not in holidays[str(day.year)]:
            return day.isoformat()
        day -= timedelta(days=1)


def completed_history(history, expected_date):
    dates = pd.to_datetime(history["日期"], errors="raise")
    frame = history.loc[dates <= pd.Timestamp(expected_date)].copy()
    if frame.empty or pd.to_datetime(frame["日期"]).max() != pd.Timestamp(expected_date):
        latest = None if frame.empty else str(pd.to_datetime(frame["日期"]).max().date())
        raise ValueError(f"日线未达到最近完整交易日 {expected_date}（实际 {latest}）；保留旧文件，需核对停牌或源延迟。")
    return frame
