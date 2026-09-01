"""可追溯的 A 股日线主备数据源。

每次请求只采用一个来源返回的完整日线，主源失败时才切换备用源；不会把不同
来源、不同复权口径的行拼接在一起。调用方可将 ``MarketDataFetchResult`` 的
来源信息写入本地审计文件，但正式 ``data/*.csv`` 仍只保存统一 OHLCV 字段。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd
import requests


HISTORY_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]


class MarketDataSourceError(ValueError):
    """所有候选数据源都未能提供有效日线。"""


class DailyHistorySource(Protocol):
    """单一日线提供方的最小接口，方便离线测试和后续增加来源。"""

    name: str
    adjustment: str

    def fetch(self, market_code: str) -> pd.DataFrame:
        """返回已规范化的 OHLCV 日线；失败时抛出异常。"""


@dataclass(frozen=True)
class MarketDataFetchResult:
    """一次成功日线请求及其最小可审计元数据。"""

    history: pd.DataFrame
    source: str
    adjustment: str
    used_fallback: bool
    attempted_sources: tuple[str, ...]
    failures: tuple[dict, ...]

    def provenance(self, stock_code: str, market_code: str) -> dict:
        latest = self.history["日期"].max()
        earliest = self.history["日期"].min()
        return {
            "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "股票代码": str(stock_code),
            "市场代码": str(market_code),
            "数据源": self.source,
            "复权方式": self.adjustment,
            "是否使用备用源": self.used_fallback,
            "尝试来源": list(self.attempted_sources),
            "失败记录": list(self.failures),
            "有效行数": int(len(self.history)),
            "日期范围": [
                earliest.strftime("%Y-%m-%d"),
                latest.strftime("%Y-%m-%d"),
            ],
            "口径": "单次更新只使用一个来源的前复权日线；不会跨来源拼接。",
        }


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化来源返回值，拒绝无效 OHLCV 或重复日期。"""
    missing = set(HISTORY_COLUMNS).difference(history.columns)
    if missing:
        raise ValueError(f"日线缺少字段：{'、'.join(sorted(missing))}。")
    normalized = history[HISTORY_COLUMNS].copy()
    normalized["日期"] = pd.to_datetime(normalized["日期"], errors="coerce")
    for column in HISTORY_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    invalid = normalized.isna().any(axis=1)
    invalid |= (normalized[["开盘", "收盘", "最高", "最低"]] <= 0).any(axis=1)
    invalid |= normalized["成交量"] < 0
    invalid |= normalized["最高"] < normalized[["开盘", "收盘", "最低"]].max(axis=1)
    invalid |= normalized["最低"] > normalized[["开盘", "收盘", "最高"]].min(axis=1)
    normalized = normalized.loc[~invalid].sort_values("日期").drop_duplicates("日期", keep="last")
    normalized = normalized.reset_index(drop=True)
    if normalized.empty:
        raise ValueError("日线没有通过 OHLCV 校验的有效记录。")
    return normalized


class TencentQfqDailySource:
    """腾讯前复权日线主源。"""

    name = "腾讯行情 qfqday"
    adjustment = "qfq"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    def __init__(self, request_get=requests.get):
        self.request_get = request_get

    def fetch(self, market_code: str) -> pd.DataFrame:
        response = self.request_get(
            self.url,
            params={"param": f"{market_code},day,,,600,qfq"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        try:
            rows = response.json()["data"][market_code]["qfqday"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"腾讯行情未返回 {market_code} 的前复权日线。") from error
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"腾讯行情未返回 {market_code} 的前复权日线。")
        frame = pd.DataFrame(rows)
        if frame.shape[1] < 6:
            raise ValueError("腾讯行情日线字段不足。")
        frame = frame.iloc[:, :6]
        frame.columns = HISTORY_COLUMNS
        return normalize_history(frame)


class EastmoneyQfqDailySource:
    """东方财富前复权日线备用源。"""

    name = "东方财富 qfq kline"
    adjustment = "qfq"
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def __init__(self, request_get=requests.get):
        self.request_get = request_get

    @staticmethod
    def _secid(market_code: str) -> str:
        code = str(market_code).strip().lower()
        if len(code) != 8 or code[:2] not in {"sh", "sz"} or not code[2:].isdigit():
            raise ValueError(f"不支持的 A 股市场代码：{market_code}。")
        return f"1.{code[2:]}" if code.startswith("sh") else f"0.{code[2:]}"

    def fetch(self, market_code: str) -> pd.DataFrame:
        response = self.request_get(
            self.url,
            params={
                "secid": self._secid(market_code),
                "klt": "101",
                "fqt": "1",
                "lmt": "600",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        try:
            rows = response.json()["data"]["klines"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"东方财富未返回 {market_code} 的前复权日线。") from error
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"东方财富未返回 {market_code} 的前复权日线。")
        values = [str(row).split(",") for row in rows]
        frame = pd.DataFrame(values)
        if frame.shape[1] < 6:
            raise ValueError("东方财富日线字段不足。")
        frame = frame.iloc[:, :6]
        frame.columns = HISTORY_COLUMNS
        return normalize_history(frame)


def default_daily_sources() -> tuple[DailyHistorySource, ...]:
    """默认保持腾讯为主、东方财富为整只股票备用的固定顺序。"""
    return TencentQfqDailySource(), EastmoneyQfqDailySource()


def fetch_daily_history(market_code: str, sources=None) -> MarketDataFetchResult:
    """按顺序尝试完整日线源，成功后立即返回，不跨来源拼接。"""
    candidates = tuple(sources or default_daily_sources())
    if not candidates:
        raise ValueError("至少需要配置一个日线数据源。")
    failures = []
    attempted = []
    for index, source in enumerate(candidates):
        source_name = getattr(source, "name", type(source).__name__)
        attempted.append(source_name)
        try:
            history = normalize_history(source.fetch(market_code))
            return MarketDataFetchResult(
                history=history,
                source=source_name,
                adjustment=getattr(source, "adjustment", "unknown"),
                used_fallback=index > 0,
                attempted_sources=tuple(attempted),
                failures=tuple(failures),
            )
        except (OSError, ValueError, requests.RequestException) as error:
            failures.append({"数据源": source_name, "原因": str(error)})
    summary = "；".join(f"{item['数据源']}：{item['原因']}" for item in failures)
    raise MarketDataSourceError(f"所有日线数据源均失败：{summary}")
