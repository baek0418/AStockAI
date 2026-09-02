"""v4.6 证据层：只整理本地快照与市场指数事实，不调用 AI 或行情接口。"""

import json
from pathlib import Path

import pandas as pd

from astock_core.analysis.fundamental_data import (
    build_industry_peer_comparison,
    build_valuation_observation,
    load_fundamental_snapshot,
    summarize_fundamental_evidence,
)
from astock_core.portfolio.portfolio_management import (
    build_investment_review,
    build_portfolio_rows,
    load_portfolio,
    summarize_portfolio,
)


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()
MARKET_FILES = {
    "沪深300": "沪深300_sh000300.csv",
    "中证1000": "中证1000_sh000852.csv",
}
INSUFFICIENT = "数据不足"


def _latest_file(directory, pattern):
    files = sorted(Path(directory).glob(pattern))
    return files[-1] if files else None


def _load_json_context(directory, pattern, label):
    file_path = _latest_file(directory, pattern)
    if not file_path:
        return {"available": False, "status": f"数据不足：未找到 {label}。", "data": None, "file": None}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "status": f"数据不足：{label} 读取失败：{error}。", "data": None, "file": str(file_path)}
    if not isinstance(data, dict):
        return {"available": False, "status": f"数据不足：{label} 格式无效。", "data": None, "file": str(file_path)}
    return {"available": True, "status": "可用", "data": data, "file": str(file_path)}


def load_latest_quant_snapshot(output_directory=PROJECT_DIRECTORY / "output"):
    """读取最新量化快照；失败转为可展示的数据不足状态。"""
    context = _load_json_context(output_directory, "quant_snapshot_*.json", "quant_snapshot")
    if context["available"] and not isinstance(context["data"].get("股票排行榜"), list):
        context.update(available=False, status="数据不足：quant_snapshot 缺少股票排行榜。", data=None)
    return context


def load_previous_quant_snapshot(output_directory, latest_snapshot_file):
    """读取当前量化快照之前最近的一份有效快照，仅用于比较研究排序。"""
    output_directory = Path(output_directory)
    latest_path = Path(latest_snapshot_file) if latest_snapshot_file else None
    candidates = sorted(output_directory.glob("quant_snapshot_*.json"))
    if latest_path:
        candidates = [path for path in candidates if path.name < latest_path.name]
    for file_path in reversed(candidates):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("股票排行榜"), list):
            return {"available": True, "data": data, "file": str(file_path)}
    return {"available": False, "data": None, "file": None}


def load_latest_daily_signal(output_directory=PROJECT_DIRECTORY / "output"):
    """读取最新日间信号；没有前日快照不视为文件失败。"""
    context = _load_json_context(output_directory, "daily_signal_*.json", "daily_signal")
    if context["available"] and not isinstance(context["data"].get("stocks"), list):
        context.update(available=False, status="数据不足：daily_signal 缺少 stocks。", data=None)
    return context


def load_latest_watchlist_snapshot(output_directory=PROJECT_DIRECTORY / "output"):
    """读取最新关注股票快照。"""
    context = _load_json_context(output_directory, "watchlist_snapshot_*.json", "watchlist_snapshot")
    if context["available"] and not isinstance(context["data"].get("stocks"), list):
        context.update(available=False, status="数据不足：watchlist_snapshot 缺少 stocks。", data=None)
    return context


def load_announcement_context(output_directory, report_date):
    """只读取与日报同日的公告快照，避免把旧公告伪装成今日信息。"""
    file_path = Path(output_directory) / f"announcement_snapshot_{report_date}.json"
    if not file_path.is_file():
        return {"available": False, "status": "数据不足：当日官方公告快照未生成。", "data": None, "file": None}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "status": f"数据不足：公告快照读取失败：{error}。", "data": None, "file": str(file_path)}
    if not isinstance(data, dict) or data.get("报告日期") != report_date or not isinstance(data.get("stocks"), list):
        return {"available": False, "status": "数据不足：公告快照格式或日期无效。", "data": None, "file": str(file_path)}
    return {"available": True, "status": "可用", "data": data, "file": str(file_path)}


def load_prediction_admission_status(output_directory=PROJECT_DIRECTORY / "output"):
    """读取最近一次预测模型准入结论，不把未验收模型当成日报信号。"""
    context = _load_json_context(
        Path(output_directory) / "prediction",
        "prediction_comparison_*.json",
        "预测模型验证报告",
    )
    if not context["available"]:
        return {
            "状态": "未接入",
            "说明": "未找到可用于日报的预测模型验证报告。",
            "生成时间": INSUFFICIENT,
        }
    conclusion = context["data"].get("结论", {})
    if not isinstance(conclusion, dict):
        return {
            "状态": "未接入",
            "说明": "预测模型验证报告缺少准入结论。",
            "生成时间": context["data"].get("生成时间", INSUFFICIENT),
        }
    admitted = conclusion.get("可接入日报或 Web") is True
    return {
        "状态": "已验收" if admitted else "未接入",
        "说明": str(conclusion.get("结论") or "预测模型尚无明确准入结论。"),
        "生成时间": context["data"].get("生成时间", INSUFFICIENT),
    }


def load_watchlist_config(watchlist_file=PROJECT_DIRECTORY / "watchlist.json"):
    """读取 watchlist 配置，仅保留启用股票，不修改文件。"""
    file_path = Path(watchlist_file)
    if not file_path.is_file():
        return {"available": False, "status": "数据不足：未找到 watchlist.json。", "stocks": []}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "status": f"数据不足：watchlist.json 读取失败：{error}。", "stocks": []}
    stocks = data.get("stocks", []) if isinstance(data, dict) else []
    if not isinstance(stocks, list):
        return {"available": False, "status": "数据不足：watchlist.json 的 stocks 无效。", "stocks": []}
    return {
        "available": True,
        "status": "可用",
        "stocks": [item for item in stocks if isinstance(item, dict) and item.get("enable", True) is not False],
    }


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _quant_quotes(quant_context):
    """从同一份量化快照提取本地收盘价，绝不为日报重新请求实时行情。"""
    quant_data = quant_context.get("data") if isinstance(quant_context, dict) else {}
    report_date = (quant_data or {}).get("快照日期", INSUFFICIENT)
    quotes = {}
    for item in (quant_data or {}).get("股票排行榜", []):
        if not isinstance(item, dict):
            continue
        code = str(item.get("股票代码", "")).strip().zfill(6)
        close = _number(item.get("收盘价"))
        if not code.isdigit() or close is None:
            continue
        quotes[code] = {
            "close": close,
            "date": str(item.get("日期") or report_date),
            "advice": item.get("建议", "数据不足"),
        }
    return quotes


def build_portfolio_report_context(portfolio_file, quant_context, daily_signal_context, report_date):
    """把本地持仓、同日快照和已保存观察条件组成日报的持仓核对事实。"""
    file_path = Path(portfolio_file)
    if not file_path.is_file():
        return {
            "状态": "未录入持仓：日报仅展示关注列表，不展示账户或持仓信息。",
            "摘要": None,
            "持仓": [],
            "核对清单": [],
            "风险持仓数": 0,
            "风险市值占比": None,
            "最新报价日期": [],
        }
    try:
        portfolio = load_portfolio(file_path)
        signal_data = daily_signal_context.get("data") if isinstance(daily_signal_context, dict) else {}
        rows = build_portfolio_rows(
            portfolio,
            _quant_quotes(quant_context),
            (signal_data or {}).get("stocks", []),
        )
    except ValueError as error:
        return {
            "状态": f"持仓账本无法核对：{error}。",
            "摘要": None,
            "持仓": [],
            "核对清单": [],
            "风险持仓数": 0,
            "风险市值占比": None,
            "最新报价日期": [],
        }
    summary = summarize_portfolio(rows, portfolio)
    review = build_investment_review(rows, (signal_data or {}).get("stocks", []))
    priced_rows = [row for row in rows if _number(row.get("当前市值")) is not None]
    total_value = sum(row["当前市值"] for row in priced_rows)
    risk_codes = {
        str(item.get("股票代码", "")).zfill(6)
        for item in (signal_data or {}).get("stocks", [])
        if isinstance(item, dict) and item.get("信号分类") in {"风险", "偏弱"}
    }
    risk_rows = [row for row in priced_rows if row.get("股票代码") in risk_codes]
    risk_value = sum(row["当前市值"] for row in risk_rows)
    dates = sorted({str(row.get("行情日期")) for row in priced_rows if row.get("行情日期")})
    missing = summary.get("缺少本地报价数", 0)
    stale = [date for date in dates if str(report_date) and date != str(report_date)]
    if not rows:
        status = "未录入股票持仓：日报仅展示关注列表。"
    elif missing:
        status = f"持仓报价不完整：{missing}/{len(rows)} 只持仓未匹配本地量化快照。"
    elif stale:
        status = "部分持仓收盘日期早于日报快照；市值仅用于本地核对，不视为实时价格。"
    else:
        status = "可核对：持仓市值与研究观察均来自本地日报快照。"
    return {
        "状态": status,
        "摘要": summary,
        "持仓": rows,
        "核对清单": review,
        "风险持仓数": len(risk_rows),
        "风险市值占比": round(risk_value / total_value * 100, 2) if total_value else None,
        "最新报价日期": dates,
    }


def _market_index_evidence(index_name, csv_file):
    if not csv_file.is_file():
        return {"名称": index_name, "数据状态": f"数据不足：未找到 {csv_file.name}。"}
    try:
        data = pd.read_csv(csv_file, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        return {"名称": index_name, "数据状态": f"数据不足：指数文件读取失败：{error}。"}
    required = {"日期", "收盘"}
    if not required.issubset(data.columns):
        return {"名称": index_name, "数据状态": "数据不足：指数文件缺少日期或收盘字段。"}
    data = data.copy()
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    data["收盘"] = pd.to_numeric(data["收盘"], errors="coerce")
    data = data.dropna(subset=["日期", "收盘"]).sort_values("日期").drop_duplicates("日期", keep="last")
    if data.empty:
        return {"名称": index_name, "数据状态": "数据不足：指数文件没有有效日线。"}
    latest = data.iloc[-1]
    result = {
        "名称": index_name,
        "数据状态": "可用" if len(data) >= 21 else "数据不足：指数日线不足 21 条。",
        "数据截至日期": latest["日期"].strftime("%Y-%m-%d"),
        "最新收盘": round(float(latest["收盘"]), 4),
        "文件": str(csv_file),
    }
    for days in (1, 5, 20):
        if len(data) <= days:
            result[f"{days}日涨跌"] = INSUFFICIENT
        else:
            prior_close = float(data.iloc[-(days + 1)]["收盘"])
            result[f"{days}日涨跌"] = round((float(latest["收盘"]) / prior_close - 1) * 100, 2)
    if len(data) < 20:
        result["20日均线"] = INSUFFICIENT
        result["位于20日均线之上"] = INSUFFICIENT
    else:
        ma20 = float(data["收盘"].tail(20).mean())
        result["20日均线"] = round(ma20, 4)
        result["位于20日均线之上"] = bool(float(latest["收盘"]) > ma20)
    return result


def load_market_context(market_directory=PROJECT_DIRECTORY / "data" / "market"):
    """计算沪深300与中证1000的已知 1/5/20 日事实，不填补缺失交易日。"""
    market_directory = Path(market_directory)
    indices = {
        name: _market_index_evidence(name, market_directory / filename)
        for name, filename in MARKET_FILES.items()
    }
    available_dates = [item.get("数据截至日期") for item in indices.values() if item.get("数据截至日期")]
    return {
        "数据状态": "可用" if len(available_dates) == len(MARKET_FILES) else "部分或全部市场数据不足",
        "数据截至日期": max(available_dates) if available_dates else INSUFFICIENT,
        "指数": indices,
    }


def _load_provenance_record(provenance_directory, stock):
    """读取单只股票的行情来源审计；缺失记录只降低可追溯性，不阻断日报。"""
    code = str(stock.get("code", "")).strip()
    name = str(stock.get("name", code or "未知股票")).strip()
    if not code:
        return {"股票代码": "", "股票名称": name, "状态": "未记录：股票代码无效。"}
    file_path = Path(provenance_directory) / f"{code}.json"
    if not file_path.is_file():
        return {"股票代码": code, "股票名称": name, "状态": "未记录：该日线由旧版流程生成或尚未更新。"}
    try:
        record = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"股票代码": code, "股票名称": name, "状态": f"不可用：来源审计读取失败：{error}。"}
    if not isinstance(record, dict):
        return {"股票代码": code, "股票名称": name, "状态": "不可用：来源审计格式无效。"}
    date_range = record.get("日期范围", [])
    last_date = date_range[-1] if isinstance(date_range, list) and date_range else INSUFFICIENT
    return {
        "股票代码": code,
        "股票名称": name,
        "状态": "可核对",
        "数据源": str(record.get("数据源") or INSUFFICIENT),
        "复权方式": str(record.get("复权方式") or INSUFFICIENT),
        "数据截至日期": str(last_date),
        "更新时间": str(record.get("更新时间") or INSUFFICIENT),
        "是否使用备用源": record.get("是否使用备用源") is True,
        "请求尝试次数": record.get("请求尝试次数", 1),
    }


def load_quote_provenance_context(data_directory, watchlist_stocks, report_date):
    """汇总本地行情审计，不以更新后的文件反向改写日报快照中的事实。"""
    records = [
        _load_provenance_record(Path(data_directory) / "provenance", stock)
        for stock in watchlist_stocks or []
    ]
    verifiable = [item for item in records if item.get("状态") == "可核对"]
    report_day = str(report_date or INSUFFICIENT)[:10]
    same_day = [item for item in verifiable if item.get("数据截至日期") == report_day]
    newer = [item for item in verifiable if item.get("数据截至日期") > report_day]
    older = [item for item in verifiable if item.get("数据截至日期") < report_day]
    missing = [item for item in records if item.get("状态") != "可核对"]
    if not records:
        status = "数据不足：没有启用关注股，无法核对行情来源。"
    elif missing or older:
        status = "审计不完整：不因缺失或落后的来源记录扩大日报结论。"
    elif newer:
        status = "当前行情记录较日报快照更新；日报仍只使用快照日期的已保存事实。"
    else:
        status = "可核对：关注股行情来源记录与日报快照日期一致。"
    return {
        "状态": status,
        "日报快照日期": report_day,
        "关注股数": len(records),
        "可核对数": len(verifiable),
        "同日记录数": len(same_day),
        "较新记录数": len(newer),
        "落后记录数": len(older),
        "未记录数": len(missing),
        "备用源更新数": sum(item.get("是否使用备用源") is True for item in verifiable),
        "重试后成功数": sum(
            isinstance(item.get("请求尝试次数"), int) and item["请求尝试次数"] > 1
            for item in verifiable
        ),
        "股票": records,
    }


def _find_stock(stocks, code, name, code_keys=("股票代码", "code"), name_keys=("股票名称", "name")):
    for item in stocks or []:
        if not isinstance(item, dict):
            continue
        if code and any(str(item.get(key, "")).strip() == str(code) for key in code_keys):
            return item
    for item in stocks or []:
        if not isinstance(item, dict):
            continue
        if name and any(str(item.get(key, "")).strip() == str(name) for key in name_keys):
            return item
    return None


def _rsi_description(rsi):
    if rsi is None:
        return None
    if rsi >= 70:
        return f"RSI 为 {rsi}，位于偏高区间"
    if rsi >= 50:
        return f"RSI 为 {rsi}，位于中性偏强区间"
    if rsi >= 30:
        return f"RSI 为 {rsi}，位于中性偏弱区间"
    return f"RSI 为 {rsi}，位于偏低区间"


def _stock_evidence_lists(current, changes, market_context):
    strong, cautious, conditions = [], [], []
    ma5, ma20 = _number(current.get("MA5")), _number(current.get("MA20"))
    macd, rsi, score = _number(current.get("MACD")), _number(current.get("RSI")), _number(current.get("Score"))
    if ma5 is not None and ma20 is not None:
        if ma5 > ma20:
            strong.append("MA5 高于 MA20。")
            conditions.append("若 MA5 持续高于 MA20，则短期趋势结构仍未破坏。")
        else:
            cautious.append("MA5 低于或等于 MA20。")
            conditions.append("若 MA5 回到 MA20 上方，再观察均线关系是否改善。")
    if macd is not None:
        if macd > 0:
            strong.append("MACD 为正。")
            conditions.append("若 MACD 继续为正，则观察动能状态是否延续。")
        else:
            cautious.append("MACD 为负或为零。")
            conditions.append("若 MACD 继续为负且评分下降，则应重点观察动能是否进一步减弱。")
    rsi_text = _rsi_description(rsi)
    if rsi_text:
        (strong if 50 <= rsi < 70 else cautious).append(rsi_text + "。")
    if score is not None:
        score_change = changes.get("Score变化")
        if isinstance(score_change, (int, float)):
            if score_change > 0:
                strong.append(f"Score 较上一交易日上升 {score_change}。")
            elif score_change < 0:
                cautious.append(f"Score 较上一交易日下降 {abs(score_change)}。")
    if "由" in str(changes.get("MA5/MA20关系变化", "")):
        cautious.append(f"均线关系发生变化：{changes['MA5/MA20关系变化']}。")
    if "转为" in str(changes.get("MACD状态变化", "")):
        cautious.append(f"MACD 状态发生切换：{changes['MACD状态变化']}。")
    indices = market_context.get("指数", {})
    if any(item.get("位于20日均线之上") is False for item in indices.values()):
        conditions.append("若市场指数同步走弱，应注意个股强势信号的可靠性下降。")
    return (
        strong or ["未发现满足规则的偏强证据。"],
        cautious or ["未发现满足规则的谨慎证据。"],
        list(dict.fromkeys(conditions)) or ["数据不足，无法设置观察条件。"],
    )


def build_stock_evidence(
    stock,
    quant_context,
    daily_signal_context,
    watchlist_snapshot_context=None,
    watchlist_config_context=None,
    market_context=None,
    announcement_context=None,
):
    """把单股当前指标、日间变化及可追溯证据组合为只读事实对象。"""
    stock = stock or {}
    code = str(stock.get("code", stock.get("股票代码", ""))).strip()
    name = str(stock.get("name", stock.get("股票名称", ""))).strip()
    quant_data = quant_context.get("data") if quant_context.get("available") else {}
    quant_item = _find_stock((quant_data or {}).get("股票排行榜", []), code, name)
    signal_data = daily_signal_context.get("data") if daily_signal_context and daily_signal_context.get("available") else {}
    signal_item = _find_stock((signal_data or {}).get("stocks", []), code, name)
    watch_data = watchlist_snapshot_context.get("data") if watchlist_snapshot_context and watchlist_snapshot_context.get("available") else {}
    watch_item = _find_stock((watch_data or {}).get("stocks", []), code, name)
    announcement_data = announcement_context.get("data") if announcement_context and announcement_context.get("available") else {}
    announcement_item = _find_stock((announcement_data or {}).get("stocks", []), code, name)
    config_stocks = (watchlist_config_context or {}).get("stocks", [])
    config_item = _find_stock(config_stocks, code, name)
    source = signal_item.get("当前指标", {}) if signal_item else {}
    if not source and quant_item:
        source = {
            "Score": quant_item.get("综合评分"), "RSI": quant_item.get("RSI"), "MA5": quant_item.get("MA5"),
            "MA20": quant_item.get("MA20"), "MACD": quant_item.get("MACD"), "趋势": "快照未提供",
            "建议": quant_item.get("建议"), "风险标签": "快照未提供",
        }
    if not source and watch_item and watch_item.get("status") != "missing":
        source = {
            "Score": watch_item.get("score"), "RSI": watch_item.get("rsi"), "MA5": watch_item.get("ma5"),
            "MA20": watch_item.get("ma20"), "MACD": watch_item.get("macd"), "趋势": watch_item.get("trend"),
            "建议": watch_item.get("advice"), "风险标签": watch_item.get("risk"),
        }
    current = {
        key: source.get(key) if source.get(key) is not None else INSUFFICIENT
        for key in ("Score", "RSI", "MA5", "MA20", "MACD", "趋势", "建议", "风险标签")
    }
    # daily_signal 只保存用于比较的技术指标；收盘价以同一日期的
    # quant_snapshot 为准，避免日报为了展示价格重新读取行情文件。
    current["收盘价"] = (
        quant_item.get("收盘价")
        if quant_item and quant_item.get("收盘价") is not None
        else INSUFFICIENT
    )
    extended = (quant_item or {}).get("扩展技术指标", {})
    current["价格结构"] = {
        "5日涨跌": _number(extended.get("momentum_5d")),
        "10日涨跌": _number(extended.get("momentum_10d")),
        "20日涨跌": _number(extended.get("momentum_20d")),
        "20日波动率": _number(extended.get("volatility_20d")),
        "相对5日成交量": _number(extended.get("volume_relative_5d")),
    }
    changes = (signal_item or {}).get("今日变化") or {
        "Score变化": INSUFFICIENT, "RSI变化": INSUFFICIENT,
        "MA5/MA20关系变化": INSUFFICIENT, "MACD状态变化": INSUFFICIENT,
    }
    market_context = market_context or {"指数": {}}
    strong, cautious, conditions = _stock_evidence_lists(current, changes, market_context)
    return {
        "股票代码": code or (quant_item or {}).get("股票代码", ""),
        "股票名称": name or (quant_item or {}).get("股票名称", "未知股票"),
        "别名": (config_item or watch_item or {}).get("alias", ""),
        "标签": (config_item or watch_item or {}).get("tags", []),
        "优先级": (config_item or watch_item or {}).get("priority", INSUFFICIENT),
        "数据状态": (signal_item or {}).get("数据状态", quant_context.get("status", f"{INSUFFICIENT}：无个股量化记录。")),
        "量化数据截至日期": (quant_data or {}).get("快照日期", INSUFFICIENT),
        "当前量化证据": current,
        "今日变化": changes,
        "信号分类": (signal_item or {}).get("信号分类", INSUFFICIENT),
        "偏强证据": strong,
        "谨慎证据": cautious,
        "观察重点": conditions,
        "近期官方公告": (announcement_item or {}).get("公告", []),
        "公告数据状态": (announcement_item or {}).get("数据状态", (announcement_context or {}).get("status", "数据不足")),
        "事实来源": {
            "quant_snapshot": quant_context.get("file"),
            "daily_signal": (daily_signal_context or {}).get("file"),
            "watchlist_snapshot": (watchlist_snapshot_context or {}).get("file"),
        },
    }


def build_market_evidence(market_context):
    """返回市场证据的稳定副本，供报告与 AI 共用。"""
    return {"市场环境": market_context}


def attach_fundamental_research(stock_evidence, fundamental_directory):
    """为日报股票附加只读基本面、估值观察和严格同业比较。"""
    enriched = dict(stock_evidence)
    try:
        snapshot = load_fundamental_snapshot(enriched.get("股票代码", ""), fundamental_directory)
        fundamental = summarize_fundamental_evidence(snapshot)
        peer_comparison = build_industry_peer_comparison(snapshot, fundamental_directory)
    except ValueError as error:
        fundamental = {"数据状态": f"数据不足：基本面快照股票代码无效：{error}。", "事实": []}
        peer_comparison = {"数据状态": "数据不足：基本面快照股票代码无效，不能进行同业比较。"}
    fundamental["价格日期"] = enriched.get("量化数据截至日期", INSUFFICIENT)
    valuation = build_valuation_observation(
        fundamental, enriched.get("当前量化证据", {}).get("收盘价")
    )
    enriched["基本面研究证据"] = fundamental
    enriched["估值观察"] = valuation
    enriched["行业同业比较"] = peer_comparison
    return enriched


def build_research_recommendations(quant_context, limit=3):
    """从当日量化快照稳定选出最多三只优先研究标的。

    这份排序独立于 20 日稳健候选的市场闸门：前者保证日报始终有可供
    阅读的三个研究优先级，后者继续只在严格条件全部通过时才出现。
    """
    quant_data = quant_context.get("data") if quant_context.get("available") else {}
    candidate_snapshot = (quant_data or {}).get("稳健研究候选", {})
    recommendations = candidate_snapshot.get("20日研究推荐") if isinstance(candidate_snapshot, dict) else None
    if isinstance(recommendations, list):
        return recommendations[:limit]
    rankings = (quant_data or {}).get("股票排行榜", [])
    if not isinstance(rankings, list):
        return []

    available = [
        stock for stock in rankings
        if isinstance(stock, dict) and _number(stock.get("综合评分")) is not None
    ]
    available.sort(
        key=lambda stock: (-_number(stock.get("综合评分")), str(stock.get("股票名称", "")))
    )

    def trend_label(stock):
        ma5, ma20 = _number(stock.get("MA5")), _number(stock.get("MA20"))
        if ma5 is None or ma20 is None:
            return "数据不足"
        if ma5 > ma20:
            return "均线多头"
        if ma5 < ma20:
            return "均线偏弱"
        return "均线持平"

    return [
        {
            "股票代码": stock.get("股票代码", ""),
            "股票名称": stock.get("股票名称", "未知股票"),
            "综合评分": stock.get("综合评分"),
            "RSI": stock.get("RSI"),
            "MA5": stock.get("MA5"),
            "MA20": stock.get("MA20"),
            "MACD": stock.get("MACD"),
            "技术趋势": stock.get("技术趋势") or trend_label(stock),
            "建议": stock.get("建议", "数据不足"),
        }
        for stock in available[:limit]
    ]


def add_research_priority_changes(recommendations, previous_quant_context):
    """补充研究 TOP3 内的相对变化，不把它误写成全市场排名。"""
    previous = build_research_recommendations(previous_quant_context) if previous_quant_context.get("available") else []
    previous_by_identity = {
        (str(stock.get("股票代码", "")), str(stock.get("股票名称", ""))): (index, stock)
        for index, stock in enumerate(previous, start=1)
    }
    enriched = []
    for current_rank, stock in enumerate(recommendations, start=1):
        item = dict(stock)
        identity = (str(item.get("股票代码", "")), str(item.get("股票名称", "")))
        previous_item = previous_by_identity.get(identity)
        if not previous_quant_context.get("available"):
            item["TOP3动态"] = "上一交易日 TOP3 不可用"
        elif not previous_item:
            item["TOP3动态"] = "新进入 TOP3"
        else:
            previous_rank, previous_stock = previous_item
            if current_rank < previous_rank:
                rank_text = f"TOP3 内上升 {previous_rank - current_rank} 位"
            elif current_rank > previous_rank:
                rank_text = f"TOP3 内下降 {current_rank - previous_rank} 位"
            else:
                rank_text = "TOP3 内排名不变"
            current_score = _number(item.get("20日研究优先评分", item.get("综合评分")))
            previous_score = _number(previous_stock.get("20日研究优先评分", previous_stock.get("综合评分")))
            if current_score is not None and previous_score is not None:
                score_change = round(current_score - previous_score, 1)
                rank_text += f"；研究优先评分 {'+' if score_change > 0 else ''}{score_change}"
            item["TOP3动态"] = rank_text
        enriched.append(item)
    return enriched


def build_report_evidence(
    output_directory=PROJECT_DIRECTORY / "output",
    market_directory=PROJECT_DIRECTORY / "data" / "market",
    watchlist_file=PROJECT_DIRECTORY / "watchlist.json",
    portfolio_file=None,
):
    """构造日报可用的完整事实包；缺失的任一来源不阻断规则报告。"""
    quant_context = load_latest_quant_snapshot(output_directory)
    previous_quant_context = load_previous_quant_snapshot(output_directory, quant_context.get("file"))
    daily_context = load_latest_daily_signal(output_directory)
    watch_snapshot_context = load_latest_watchlist_snapshot(output_directory)
    prediction_admission = load_prediction_admission_status(output_directory)
    quant_date = (quant_context.get("data") or {}).get("快照日期")
    announcement_context = load_announcement_context(output_directory, quant_date) if quant_date else {
        "available": False, "status": "数据不足：量化快照日期缺失，未读取公告。", "data": None, "file": None
    }
    signal_date = (daily_context.get("data") or {}).get("快照日期")
    if quant_date and signal_date and quant_date != signal_date:
        daily_context = {
            "available": False,
            "status": f"数据不足：daily_signal 日期 {signal_date} 与量化快照日期 {quant_date} 不一致。",
            "data": None,
            "file": daily_context.get("file"),
        }
    watch_config_context = load_watchlist_config(watchlist_file)
    market_context = load_market_context(market_directory)
    quote_provenance = load_quote_provenance_context(
        Path(market_directory).parent,
        watch_config_context.get("stocks", []),
        quant_date,
    )
    raw_stocks = [
        build_stock_evidence(stock, quant_context, daily_context, watch_snapshot_context, watch_config_context, market_context, announcement_context)
        for stock in watch_config_context.get("stocks", [])
    ]
    fundamental_directory = Path(market_directory).parent / "fundamentals"
    stocks = [attach_fundamental_research(stock, fundamental_directory) for stock in raw_stocks]
    portfolio_context = build_portfolio_report_context(
        portfolio_file or Path(market_directory).parent / "portfolio.json",
        quant_context,
        daily_context,
        quant_date,
    )
    return {
        "报告日期": (quant_context.get("data") or {}).get("快照日期", INSUFFICIENT),
        "量化快照": quant_context,
        "daily_signal": daily_context,
        "watchlist_snapshot": watch_snapshot_context,
        "公告证据": announcement_context,
        "预测模型验证": prediction_admission,
        "市场环境": market_context,
        "行情来源审计": quote_provenance,
        "持仓核对": portfolio_context,
        "关注股票": stocks,
        "稳健研究候选": (quant_context.get("data") or {}).get("稳健研究候选"),
        "优先研究标的": add_research_priority_changes(
            build_research_recommendations(quant_context), previous_quant_context
        ),
    }
