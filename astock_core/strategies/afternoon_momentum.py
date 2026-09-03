"""A 股午后强势筛选：只在用户明确运行时请求公开行情并保存审计快照。"""

import json
import os
import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from astock_core.data.stock_universe import create_market_code


STRATEGY_ID = "a_share_afternoon_momentum"
SKILL_NAME = "a-share-afternoon-momentum-screen"
STRATEGY_TITLE = "A股午后强势筛选"
QUOTE_URL = "https://qt.gtimg.cn/q="
KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"
CSI300_CODE = "sh000300"
QUOTE_BATCH_SIZE = 400


def strategy_catalog():
    """返回 UI 可展示的策略卡；执行逻辑由本模块而非 SKILL.md 承担。"""
    return [{
        "id": STRATEGY_ID,
        "名称": STRATEGY_TITLE,
        "周期标签": "短线",
        "观察周期": "1–5 个交易日",
        "研究风格": "动量",
        "关联 Skill": f"${SKILL_NAME}",
        "运行窗口": "A股交易日 14:30 后",
        "调仓口径": "只生成模拟观察项；模拟建仓需用户再次确认。",
        "数据要求": "全市场实时行情、日线和当日分时数据",
        "状态": "可运行",
    }]


def filter_strategy_catalog(period=None):
    """按用户选择的持有/观察周期筛选可执行策略目录。"""
    strategies = strategy_catalog()
    if not period or period == "全部":
        return strategies
    return [item for item in strategies if item.get("周期标签") == period]


def _number(fields, index):
    try:
        return float(fields[index])
    except (IndexError, TypeError, ValueError):
        return None


def parse_quote_payload(payload):
    """解析腾讯公开行情的批量文本，忽略字段不完整的记录。"""
    quotes = {}
    for match in re.finditer(r'v_([^=]+)="([^"]*)";?', str(payload)):
        market_code, raw_fields = match.groups()
        fields = raw_fields.split("~")
        if len(fields) < 52:
            continue
        quote = {
            "market_code": market_code.lower(),
            "code": str(fields[2]).strip().zfill(6),
            "name": str(fields[1]).strip(),
            "price": _number(fields, 3),
            "timestamp": str(fields[30]).strip(),
            "change_pct": _number(fields, 32),
            "high": _number(fields, 33),
            "turnover_pct": _number(fields, 38),
            "float_mcap_yi": _number(fields, 44),
            "volume_ratio": _number(fields, 49),
            "vwap": _number(fields, 51),
        }
        if quote["code"].isdigit() and quote["name"]:
            quotes[quote["market_code"]] = quote
    return quotes


def initial_filter(quote):
    """执行前四项和当前在分时均价线之上的硬条件。"""
    required = ("price", "change_pct", "turnover_pct", "float_mcap_yi", "volume_ratio", "vwap")
    if not all(isinstance(quote.get(name), (int, float)) for name in required):
        return False
    return (
        3.0 <= quote["change_pct"] <= 5.0
        and quote["volume_ratio"] > 1.0
        and 5.0 <= quote["turnover_pct"] <= 10.0
        and 50.0 <= quote["float_mcap_yi"] <= 200.0
        and quote["price"] >= quote["vwap"]
    )


def _daily_rows(payload, market_code):
    rows = payload.get("data", {}).get(market_code, {}).get("qfqday", [])
    cleaned = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            cleaned.append({"date": str(row[0]), "close": float(row[2]), "volume": float(row[5])})
        except (TypeError, ValueError):
            continue
    return cleaned


def daily_evidence(payload, market_code):
    """将图片里的量能、均线要求转换成严格且可复现的日线判定。"""
    rows = _daily_rows(payload, market_code)
    if len(rows) < 20:
        return {"available": False, "reason": "日线不足20根，无法验证均线与量能结构。"}
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows]
    ma5 = sum(closes[-5:]) / 5
    ma10 = sum(closes[-10:]) / 10
    ma20 = sum(closes[-20:]) / 20
    return {
        "available": True,
        "date": rows[-1]["date"],
        "ma5": round(ma5, 3),
        "ma10": round(ma10, 3),
        "ma20": round(ma20, 3),
        "last3_volume": [round(value, 0) for value in volumes[-3:]],
        "volume_staircase": volumes[-1] > volumes[-2] > volumes[-3],
        "ma_bull": closes[-1] > ma5 > ma10 > ma20,
    }


def minute_evidence(payload, market_code, index_change_pct):
    """核验全天 VWAP、相对指数强弱与 14:30 形态，不以日线替代分时。"""
    raw_rows = payload.get("data", {}).get(market_code, {}).get("data", {}).get("data", [])
    rows = []
    for raw in raw_rows:
        parts = str(raw).split()
        if len(parts) < 4:
            continue
        try:
            minute, price, volume, amount = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if volume > 0:
            rows.append({"minute": minute, "price": price, "vwap": amount / (volume * 100)})
    if not rows:
        return {"available": False, "reason": "未取得有效当日分时数据。"}
    session_high = max(row["price"] for row in rows)
    high_rows = [row for row in rows if row["price"] == session_high]
    first_high = high_rows[0]["minute"]
    window_rows = [row for row in rows if "1420" <= row["minute"] <= "1440"]
    after_high = [row for row in rows if row["minute"] >= first_high]
    all_at_or_above_vwap = all(row["price"] >= row["vwap"] - 0.000001 for row in rows)
    post_high_at_or_above_vwap = all(row["price"] >= row["vwap"] - 0.000001 for row in after_high)
    retraced_to_vwap = any(
        0 <= (row["price"] - row["vwap"]) / row["vwap"] <= 0.01
        for row in after_high if row["vwap"] > 0
    )
    at_1430 = next((row for row in rows if row["minute"] == "1430"), None)
    return {
        "available": True,
        "all_at_or_above_vwap": all_at_or_above_vwap,
        "session_high": session_high,
        "session_high_time": first_high,
        "near_1430_new_high": bool(window_rows) and max(row["price"] for row in window_rows) == session_high,
        "post_high_at_or_above_vwap": post_high_at_or_above_vwap,
        "retraced_to_vwap": retraced_to_vwap,
        "price_at_1430": at_1430["price"] if at_1430 else None,
        "vwap_at_1430": round(at_1430["vwap"], 4) if at_1430 else None,
        "index_change_pct": index_change_pct,
    }


def evaluate_candidate(quote, daily, minute, index_change_pct):
    """合并每条条件的证据，保证近似项不会混入严格候选。"""
    failures = []
    if not daily.get("available"):
        failures.append(daily.get("reason", "日线数据不足。"))
    else:
        if not daily["volume_staircase"]:
            failures.append("最近3根日线成交量未严格递增。")
        if not daily["ma_bull"]:
            failures.append("未满足收盘价 > MA5 > MA10 > MA20。")
    if not minute.get("available"):
        failures.append(minute.get("reason", "分时数据不足。"))
    else:
        if not minute["all_at_or_above_vwap"]:
            failures.append("全天存在跌破分时均价线/VWAP的分钟。")
        if index_change_pct is None or quote["change_pct"] <= index_change_pct:
            failures.append("未能验证当日表现强于沪深300。")
        if not minute["near_1430_new_high"]:
            failures.append("14:20–14:40未创当日新高。")
        if not minute["post_high_at_or_above_vwap"]:
            failures.append("创高后存在跌破分时均价线/VWAP的分钟。")
        if not minute["retraced_to_vwap"]:
            failures.append("创高后未验证回踩至VWAP上方1%以内。")
    return {
        "股票代码": quote["code"],
        "股票名称": quote["name"],
        "数据时间": quote["timestamp"],
        "涨幅(%)": quote["change_pct"],
        "量比": quote["volume_ratio"],
        "换手率(%)": quote["turnover_pct"],
        "流通市值(亿元)": quote["float_mcap_yi"],
        "现价": quote["price"],
        "分时均价/VWAP": quote["vwap"],
        "日线证据": daily,
        "分时证据": minute,
        "通过": not failures,
        "未通过条件": failures,
    }


def _request_text(request_get, url, **kwargs):
    response = request_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, **kwargs)
    response.raise_for_status()
    return response.content.decode("gbk", errors="replace")


def _request_json(request_get, url, **kwargs):
    response = request_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, **kwargs)
    response.raise_for_status()
    return response.json()


def run_afternoon_momentum_screen(catalog, request_get=requests.get, now=None):
    """执行一次全沪深市场午后筛选；网络失败不会用旧快照替代实时条件。"""
    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    if current.weekday() >= 5 or current.time() < time(14, 30):
        return {"status": "not_ready", "message": "策略仅在A股交易日14:30后运行。", "retrieved_at": current.isoformat()}
    market_codes = []
    for stock in catalog:
        try:
            market_codes.append(create_market_code(stock.get("code", "")))
        except (AttributeError, ValueError):
            continue
    market_codes = list(dict.fromkeys(market_codes))
    if len(market_codes) < 3000:
        return {"status": "failed", "message": "本地A股代码目录不完整，无法进行全市场筛选。", "retrieved_at": current.isoformat()}
    try:
        quote_texts = []
        for start in range(0, len(market_codes), QUOTE_BATCH_SIZE):
            quote_texts.append(_request_text(request_get, QUOTE_URL + ",".join(market_codes[start:start + QUOTE_BATCH_SIZE])))
        quotes = parse_quote_payload("\n".join(quote_texts))
    except requests.RequestException as error:
        return {"status": "failed", "message": f"实时行情请求失败：{error}。", "retrieved_at": current.isoformat()}
    # quote_map 以上必须逐批解析，避免将行情文本误当成字典。
    if not quotes:
        return {"status": "failed", "message": "实时行情未返回可解析的全市场记录。", "retrieved_at": current.isoformat()}
    index_quote = {}
    try:
        index_quote = next(iter(parse_quote_payload(_request_text(request_get, QUOTE_URL + CSI300_CODE)).values()), {})
        index_change_pct = index_quote.get("change_pct")
    except requests.RequestException:
        index_change_pct = None
    initial = [quote for quote in quotes.values() if initial_filter(quote)]
    evaluated = []
    for quote in initial:
        try:
            daily_payload = _request_json(request_get, KLINE_URL, params={"param": f"{quote['market_code']},day,,,60,qfq"})
            minute_payload = _request_json(request_get, MINUTE_URL, params={"code": quote["market_code"]})
            evaluated.append(evaluate_candidate(
                quote,
                daily_evidence(daily_payload, quote["market_code"]),
                minute_evidence(minute_payload, quote["market_code"], index_change_pct),
                index_change_pct,
            ))
        except (requests.RequestException, ValueError, TypeError) as error:
            evaluated.append({
                "股票代码": quote["code"], "股票名称": quote["name"], "数据时间": quote["timestamp"],
                "通过": False, "未通过条件": [f"日线或分时数据请求失败：{error}。"],
            })
    candidates = sorted((row for row in evaluated if row["通过"]), key=lambda row: (-row["量比"], -row["涨幅(%)"]))[:5]
    near_misses = sorted((row for row in evaluated if not row["通过"]), key=lambda row: (len(row["未通过条件"]), -row.get("量比", 0)))[:3]
    return {
        "status": "success",
        "message": "筛选完成。" if candidates else "今日无完全符合条件的候选。",
        "strategy_id": STRATEGY_ID,
        "strategy_name": STRATEGY_TITLE,
        "skill_name": SKILL_NAME,
        "retrieved_at": current.isoformat(),
        "universe": "沪深A股",
        "universe_count": len(market_codes),
        "initial_count": len(initial),
        "index": {"name": "沪深300", "change_pct": index_change_pct, "timestamp": index_quote.get("timestamp") if index_change_pct is not None else None},
        "candidates": candidates,
        "near_misses": near_misses,
        "data_source": "腾讯公开行情、日线与分时接口",
    }


def save_strategy_run(result, output_directory):
    """原子保存每次筛选的审计结果，不覆盖用户其他研究快照。"""
    directory = Path(output_directory) / "strategy_runs"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = str(result.get("retrieved_at", "")).replace(":", "").replace("+", "_").replace("-", "")[:15]
    target = directory / f"{STRATEGY_ID}_{timestamp or 'unknown'}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_latest_strategy_run(output_directory):
    directory = Path(output_directory) / "strategy_runs"
    files = sorted(directory.glob(f"{STRATEGY_ID}_*.json"), reverse=True) if directory.is_dir() else []
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
