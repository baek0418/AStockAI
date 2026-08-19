"""20 个交易日稳健研究候选：仅使用截至报告日可见的本地日线事实。"""

from pathlib import Path

import pandas as pd

from prediction_features import load_history_csv


HOLDING_DAYS = 20
MIN_HISTORY_DAYS = 60
MIN_TECHNICAL_SCORE = 65
RSI_LOWER_BOUND = 45
RSI_UPPER_BOUND = 68
MAX_20D_RETURN = 0.15
MAX_DAILY_VOLATILITY = 0.035
MIN_VOLUME_RATIO = 0.80
MAX_CANDIDATES = 3
MARKET_FILE = "沪深300_sh000300.csv"


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _read_market_history(market_directory, as_of_date):
    """读取基准指数到报告日，拒绝使用报告日之后的数据。"""
    file_path = Path(market_directory) / MARKET_FILE
    if not file_path.is_file():
        return None, f"未找到市场基准文件 {MARKET_FILE}。"
    try:
        history = load_history_csv(file_path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return None, f"市场基准数据不可用：{error}"
    history = history[history["日期"] <= pd.Timestamp(as_of_date)].copy()
    if len(history) < MIN_HISTORY_DAYS:
        return None, "市场基准历史不足 60 个交易日。"
    if history.iloc[-1]["日期"].date().isoformat() != str(as_of_date):
        return None, "市场基准与报告日期不一致，无法确认当天市场环境。"
    return history, None


def assess_market_gate(market_directory, as_of_date):
    """稳健候选的市场门槛：指数必须处于中期趋势之上且近 20 日不为负。"""
    history, error = _read_market_history(market_directory, as_of_date)
    if error:
        return {"passed": False, "reason": error}
    close = history["收盘"]
    latest = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    return20 = float(latest / close.iloc[-21] - 1)
    if latest <= ma20:
        return {
            "passed": False,
            "reason": "沪深300低于近 20 个交易日平均价格，整体市场环境偏弱。",
            "20日涨跌": round(return20 * 100, 2),
        }
    if return20 <= 0:
        return {
            "passed": False,
            "reason": "沪深300近 20 个交易日仍未上涨，稳健筛选暂不开放。",
            "20日涨跌": round(return20 * 100, 2),
        }
    return {
        "passed": True,
        "reason": "沪深300位于近 20 个交易日平均价格之上，且近 20 日为上涨。",
        "20日涨跌": round(return20 * 100, 2),
    }


def _stock_metrics(history, as_of_date):
    data = history[history["日期"] <= pd.Timestamp(as_of_date)].copy()
    if len(data) < MIN_HISTORY_DAYS:
        return None, "历史日线不足 60 个交易日。"
    if data.iloc[-1]["日期"].date().isoformat() != str(as_of_date):
        return None, "股票数据日期与报告日期不一致。"

    close = data["收盘"]
    volume = data["成交量"]
    ma5 = float(close.tail(5).mean())
    ma20 = float(close.tail(20).mean())
    return20 = float(close.iloc[-1] / close.iloc[-21] - 1)
    volatility = float(close.pct_change().tail(20).std())
    volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean())
    macd_series = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    return {
        "收盘价": round(float(close.iloc[-1]), 2),
        "MA5": round(ma5, 2),
        "MA20": round(ma20, 2),
        "20日涨跌": round(return20 * 100, 2),
        "20日波动率": round(volatility * 100, 2),
        "成交量相对20日均值": round(volume_ratio, 2),
        "MACD": round(float(macd_series.iloc[-1]), 4),
        "MACD变化": round(float(macd_series.iloc[-1] - macd_series.iloc[-2]), 4),
    }, None


def calculate_research_priority_score(score, rsi, macd, metrics):
    """计算固定的 20 日研究优先评分；仅用于横向研究排序。"""
    rsi_fit = max(0, 1 - abs(rsi - 55) / 25) if rsi is not None else 0
    trend_fit = 1 if metrics["MA5"] > metrics["MA20"] and metrics["收盘价"] >= metrics["MA20"] else 0
    momentum_fit = 1 if macd is not None and macd > 0 and metrics["MACD变化"] > 0 else 0
    return_fit = 1 if 0 < metrics["20日涨跌"] / 100 <= MAX_20D_RETURN else 0
    volatility_fit = max(0, 1 - metrics["20日波动率"] / (MAX_DAILY_VOLATILITY * 100))
    volume_fit = min(1, metrics["成交量相对20日均值"] / MIN_VOLUME_RATIO)
    return round(
        max(0, min(score or 0, 100)) * 0.35
        + rsi_fit * 20
        + trend_fit * 15
        + momentum_fit * 10
        + return_fit * 10
        + volatility_fit * 5
        + volume_fit * 5,
        1,
    )


def assess_stock_candidate(stock, data_directory, as_of_date):
    """对单股逐条执行稳健筛选，并返回可展示的通过/排除理由。"""
    score = _number(stock.get("综合评分"))
    rsi = _number(stock.get("RSI"))
    macd = _number(stock.get("MACD"))
    file_name = stock.get("数据文件")
    if not file_name:
        return {"passed": False, "股票名称": stock.get("股票名称", "未知股票"), "reason": "缺少对应历史数据文件。"}
    try:
        history = load_history_csv(Path(data_directory) / file_name)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return {"passed": False, "股票名称": stock.get("股票名称", "未知股票"), "reason": f"历史数据不可用：{error}"}
    metrics, error = _stock_metrics(history, as_of_date)
    if error:
        return {"passed": False, "股票名称": stock.get("股票名称", "未知股票"), "reason": error}

    failures = []
    if score is None or score < MIN_TECHNICAL_SCORE:
        failures.append(f"技术条件评分低于 {MIN_TECHNICAL_SCORE} 分")
    if rsi is None or not RSI_LOWER_BOUND <= rsi <= RSI_UPPER_BOUND:
        failures.append(f"短期价格强弱不在 {RSI_LOWER_BOUND}–{RSI_UPPER_BOUND} 的稳健区间")
    if metrics["MA5"] <= metrics["MA20"] or metrics["收盘价"] < metrics["MA20"]:
        failures.append("短期走势尚未稳定强于中期走势")
    if macd is None or macd <= 0 or metrics["MACD变化"] <= 0:
        failures.append("短期向上动能不足或正在减弱")
    if not 0 < metrics["20日涨跌"] / 100 <= MAX_20D_RETURN:
        failures.append("近 20 日涨幅不在稳健区间")
    if metrics["20日波动率"] / 100 > MAX_DAILY_VOLATILITY:
        failures.append("近 20 日的日度波动偏高")
    if metrics["成交量相对20日均值"] < MIN_VOLUME_RATIO:
        failures.append("当天成交活跃度低于近 20 日常态")

    base = {
        "股票名称": stock.get("股票名称", "未知股票"),
        "股票代码": stock.get("股票代码", ""),
        "综合评分": score,
        "RSI": rsi,
        "短期价格强弱值": rsi,
        **metrics,
    }
    # 20 日研究优先级并不等同于买入资格。它让市场处于防守状态时，日报仍
    # 能给出三个最值得继续核查的标的，而严格候选仍坚持全部门槛。
    priority_score = calculate_research_priority_score(score, rsi, macd, metrics)
    base["20日研究优先评分"] = priority_score
    if failures:
        return {
            "passed": False,
            **base,
            "reason": "；".join(failures) + "。",
            "未满足条件": failures,
        }

    # 仅用于通过筛选后的候选排序，不代表未来收益概率或买卖建议。
    return {
        "passed": True,
        **base,
        "稳健研究评分": priority_score,
        "入选原因": [
            "技术条件评分达到稳健门槛。",
            "短期走势高于中期走势，且收盘价未跌破中期平均水平。",
            "短期向上动能仍在增强。",
            "近 20 日涨幅、波动和成交活跃度均在稳健筛选范围内。",
        ],
        "风险提示": [
            "该筛选只使用历史价格与成交量，不包含基本面、公告、估值或盘中信息。",
            "即使通过筛选，20 个交易日内仍可能亏损；不构成买入建议。",
        ],
    }


def build_conservative_candidates(stock_rankings, data_directory, market_directory, as_of_date):
    """生成严格候选和固定三只的 20 日研究优先级。"""
    market_gate = assess_market_gate(market_directory, as_of_date)
    result = {
        "策略名称": "20 个交易日稳健研究候选",
        "报告日期": str(as_of_date),
        "持有期（交易日）": HOLDING_DAYS,
        "市场环境": market_gate,
        "候选股票": [],
        "20日研究推荐": [],
        "排除记录": [],
        "说明": "候选仅用于研究排序，不构成投资建议；筛选不通过时不会为了凑足三只而放宽条件。",
    }
    assessments = [
        assess_stock_candidate(stock, data_directory, as_of_date)
        for stock in stock_rankings
    ]
    ranked = [item for item in assessments if "20日研究优先评分" in item]
    ranked.sort(key=lambda item: (-item["20日研究优先评分"], item["股票名称"]))
    for item in ranked[:MAX_CANDIDATES]:
        recommendation = dict(item)
        if not market_gate["passed"]:
            recommendation["推荐状态"] = "市场防守观察：暂不视为新增持仓候选"
        elif item["passed"]:
            recommendation["推荐状态"] = "符合 20 日稳健候选条件"
        else:
            recommendation["推荐状态"] = "个股条件待改善，暂不视为稳健候选"
        result["20日研究推荐"].append(recommendation)

    passed = [item for item in assessments if item["passed"]]
    passed.sort(key=lambda item: (-item["稳健研究评分"], item["股票名称"]))
    if market_gate["passed"]:
        result["候选股票"] = passed[:MAX_CANDIDATES]
    result["排除记录"] = [
        {"股票名称": item["股票名称"], "原因": item["reason"]}
        for item in assessments if not item["passed"]
    ]
    if not market_gate["passed"]:
        result["说明"] = "今日未给出稳健研究候选：" + market_gate["reason"]
    elif not result["候选股票"]:
        result["说明"] = "今日没有股票同时满足 20 日稳健研究候选的全部条件。"
    return result
