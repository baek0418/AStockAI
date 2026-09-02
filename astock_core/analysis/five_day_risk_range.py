"""把已验证的 5 日风险范围转为单股票页面可解释的本地展示数据。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COVERAGE = 0.80
NORMAL_80_PERCENT_Z = 1.281552
HISTORY_DAYS = 20
SELECTED_METHOD = "历史波动率风险范围"


def _load_latest_validated_report(research_directory):
    directory = Path(research_directory)
    reports = sorted(directory.glob("return_interval_5d_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for report_file in reports:
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recommendation = report.get("推荐区间方法", {}) if isinstance(report, dict) else {}
        audit = report.get("数据可得性审计", {}) if isinstance(report, dict) else {}
        metrics = recommendation.get("指标", {}) if isinstance(recommendation, dict) else {}
        if (
            recommendation.get("方法") == SELECTED_METHOD
            and audit.get("严格时点价格依据可用") is True
            and isinstance(metrics.get("覆盖率"), (int, float))
        ):
            return report, report_file
    return None, None


def _load_local_close(history_file):
    try:
        history = pd.read_csv(history_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        history = pd.read_csv(history_file)
    required = {"日期", "收盘"}
    if required.difference(history.columns):
        raise ValueError("原始日线缺少日期或收盘字段。")
    history = history[["日期", "收盘"]].copy()
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    history["收盘"] = pd.to_numeric(history["收盘"], errors="coerce")
    history = history.dropna().sort_values("日期").drop_duplicates("日期", keep="last")
    history = history.loc[history["收盘"] > 0].reset_index(drop=True)
    if len(history) < HISTORY_DAYS + 1:
        raise ValueError(f"原始日线不足 {HISTORY_DAYS + 1} 个交易日。")
    return history


def build_five_day_risk_range(stock_record, raw_data_directory, research_directory):
    """返回用户可读的本地 5 日风险范围；缺数据时只返回原因，不猜测或联网补数。"""
    stock = stock_record if isinstance(stock_record, dict) else {}
    name = str(stock.get("name") or stock.get("股票名称") or "").strip()
    code = str(stock.get("code") or stock.get("股票代码") or "").strip().zfill(6)
    if not name or not code.isdigit() or len(code) != 6:
        return {"状态": "数据不足", "说明": "股票标识不完整，暂不能计算风险范围。"}
    report, report_file = _load_latest_validated_report(research_directory)
    if report is None:
        return {"状态": "数据不足", "说明": "尚无通过数据审计的 5 日风险范围验证报告。"}
    history_file = Path(raw_data_directory) / f"{name}历史.csv"
    if not history_file.is_file():
        return {"状态": "数据不足", "说明": "该股票尚未纳入本地原始日线快照。"}
    try:
        history = _load_local_close(history_file)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return {"状态": "数据不足", "说明": f"无法使用本地原始日线：{error}"}

    close = float(history["收盘"].iloc[-1])
    data_date = history["日期"].iloc[-1].strftime("%Y-%m-%d")
    volatility = float(history["收盘"].pct_change().rolling(HISTORY_DAYS).std().iloc[-1])
    if not np.isfinite(volatility) or volatility <= 0:
        return {"状态": "数据不足", "说明": "最近 20 个交易日的波动数据不足，暂不能计算风险范围。"}
    return_radius = NORMAL_80_PERCENT_Z * volatility * np.sqrt(5)
    lower = close * (1 - return_radius)
    upper = close * (1 + return_radius)
    metrics = report["推荐区间方法"]["指标"]
    return {
        "状态": "可用",
        "股票代码": code,
        "股票名称": name,
        "本地收盘": round(close, 3),
        "数据日期": data_date,
        "下限价格": round(float(lower), 3),
        "上限价格": round(float(upper), 3),
        "涨跌范围": round(float(return_radius * 100), 2),
        "历史覆盖率": round(float(metrics["覆盖率"]) * 100, 2),
        "验证样本数": int(metrics.get("样本数", 0)),
        "说明": f"按最近 {HISTORY_DAYS} 个交易日的波动估算未来 5 个交易日的常见收盘范围。",
        "边界": "这是风险范围，不是目标价、收益承诺或买卖建议。",
        "报告文件": str(report_file),
        "原始日线文件": str(history_file),
    }
