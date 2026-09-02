"""可复用、仅依赖当日及过去 OHLCV 的技术指标计算。

该模块不产生交易建议，也不修改预测模型特征集合。它供事实快照和后续独立研究
实验共用；所有滚动窗口与指数平滑均按日期升序计算，避免未来数据泄漏。
"""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"日期", "开盘", "收盘", "最高", "最低", "成交量"}
TECHNICAL_INDICATOR_COLUMNS = [
    "rsi_14",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "boll_middle",
    "boll_upper",
    "boll_lower",
    "boll_position",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "volatility_20d",
    "volume_relative_5d",
]


def normalize_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    """返回按日期升序、数值有效的 OHLCV，输入不足时给出明确原因。"""
    missing = REQUIRED_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"技术指标缺少字段：{'、'.join(sorted(missing))}。")
    frame = history[["日期", "开盘", "收盘", "最高", "最低", "成交量"]].copy()
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("没有可用于技术指标的有效 OHLCV 日线。")
    return frame


def calculate_technical_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """计算逐日指标；每一行的值仅使用该行及之前的 OHLCV 数据。"""
    frame = normalize_ohlcv(history)
    close = frame["收盘"]
    high = frame["最高"]
    low = frame["最低"]
    volume = frame["成交量"]
    returns = close.pct_change()

    delta = close.diff()
    gains = delta.where(delta > 0, 0.0).rolling(14).mean()
    losses = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    frame["rsi_14"] = 100 - 100 / (1 + gains / losses.replace(0, np.nan))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    frame["macd_dif"] = ema_fast - ema_slow
    frame["macd_dea"] = frame["macd_dif"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = 2 * (frame["macd_dif"] - frame["macd_dea"])

    lowest_low = low.rolling(9).min()
    highest_high = high.rolling(9).max()
    rsv = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    frame["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    frame["kdj_d"] = frame["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    frame["kdj_j"] = 3 * frame["kdj_k"] - 2 * frame["kdj_d"]

    frame["boll_middle"] = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    frame["boll_upper"] = frame["boll_middle"] + 2 * rolling_std
    frame["boll_lower"] = frame["boll_middle"] - 2 * rolling_std
    frame["boll_position"] = (close - frame["boll_lower"]) / (
        frame["boll_upper"] - frame["boll_lower"]
    ).replace(0, np.nan)

    for days in (5, 10, 20):
        frame[f"momentum_{days}d"] = close.pct_change(days)
    frame["volatility_20d"] = returns.rolling(20).std()
    frame["volume_relative_5d"] = volume / volume.rolling(5).mean().replace(0, np.nan)
    frame[TECHNICAL_INDICATOR_COLUMNS] = frame[TECHNICAL_INDICATOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return frame[["日期", *TECHNICAL_INDICATOR_COLUMNS]]


def latest_technical_indicators(history: pd.DataFrame) -> dict:
    """返回最新日的非空指标，数据窗口不足时保留可计算字段。"""
    indicators = calculate_technical_indicators(history)
    latest = indicators.iloc[-1]
    result = {"指标日期": latest["日期"].strftime("%Y-%m-%d")}
    for column in TECHNICAL_INDICATOR_COLUMNS:
        if pd.notna(latest[column]):
            result[column] = round(float(latest[column]), 6)
    return result
