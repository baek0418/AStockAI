"""AStockAI v4.5：隔离的任意 A 股按需日线下载与分析。"""

import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from daily_signal import create_signal_stock
from score import calculate_score_dataframe
from stock_universe import create_market_code, normalize_stock_code
from update_data import create_history_dataframe, get_stock


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
ON_DEMAND_DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "on_demand"
ON_DEMAND_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "on_demand"
CATALOG_FILE = ON_DEMAND_DATA_DIRECTORY / "a_share_catalog.json"
QUOTE_SOURCE = "腾讯财经单股识别接口"
MARKET_DATA_SOURCE = "腾讯财经前复权日线接口"
LOCAL_CATALOG_SOURCE = "本地 A 股代码名称目录缓存"
EASTMONEY_CATALOG_SOURCE = "东方财富公开 A 股代码名称目录接口"
EASTMONEY_CATALOG_URL = "https://push2.eastmoney.com/api/qt/clist/get"
CATALOG_PAGE_SIZE = 500
MIN_A_SHARE_CATALOG_SIZE = 3000
CATALOG_MARKET_FILTER = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"


def safe_stock_name(stock_name):
    """生成仅用于隔离缓存文件名的安全股票名称。"""
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "_", str(stock_name).strip()) or "未知股票"


def get_history_file(stock, data_directory=ON_DEMAND_DATA_DIRECTORY):
    """返回按需股票专属历史 CSV 路径。"""
    return Path(data_directory) / f"{stock['code']}_{safe_stock_name(stock['name'])}_历史.csv"


def get_snapshot_file(stock_code, output_directory=ON_DEMAND_OUTPUT_DIRECTORY):
    """返回按需分析快照路径，不与正式量化快照混用。"""
    return Path(output_directory) / f"on_demand_{normalize_stock_code(stock_code)}.json"


def read_json(json_file, description):
    """读取本地 JSON 文件。"""
    try:
        return json.loads(Path(json_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取{description}：{error}。") from error


def write_json_atomically(json_file, data):
    """原子保存 JSON，避免下载或分析中断留下空快照。"""
    target = Path(json_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_catalog(catalog_file=CATALOG_FILE):
    """读取本地代码目录缓存；未创建时返回空目录而不联网。"""
    path = Path(catalog_file)
    if not path.is_file():
        return []
    catalog_data = read_json(path, "A 股代码目录")
    stocks = catalog_data.get("stocks", []) if isinstance(catalog_data, dict) else []
    if not isinstance(stocks, list):
        raise ValueError("A 股代码目录格式错误。")
    return normalize_catalog(stocks)


def normalize_catalog(stocks):
    """清理并按代码去重目录记录。"""
    normalized = []
    seen_codes = set()
    for stock in stocks:
        if not isinstance(stock, dict):
            continue
        try:
            code = normalize_stock_code(stock.get("code", ""))
        except ValueError:
            continue
        name = str(stock.get("name", "")).strip()
        if not name or code in seen_codes:
            continue
        normalized.append({"code": code, "name": name})
        seen_codes.add(code)
    return sorted(normalized, key=lambda item: item["code"])


def merge_catalog_entries(new_entries, catalog_file=CATALOG_FILE):
    """将显式识别到的股票加入本地目录缓存，不影响 watchlist。"""
    existing_entries = load_catalog(catalog_file)
    existing_source = LOCAL_CATALOG_SOURCE
    if Path(catalog_file).is_file():
        try:
            existing_source = read_json(catalog_file, "A 股代码目录").get(
                "目录来源", existing_source
            )
        except ValueError:
            pass
    merged = {entry["code"]: entry for entry in existing_entries}
    for entry in normalize_catalog(new_entries):
        merged[entry["code"]] = entry
    merged_entries = sorted(merged.values(), key=lambda item: item["code"])
    write_json_atomically(
        catalog_file,
        {"目录来源": existing_source, "stocks": merged_entries},
    )
    return merged_entries


def _catalog_request_params(page_number):
    """构造单页 A 股名称目录请求参数。"""
    return {
        "pn": page_number,
        "pz": CATALOG_PAGE_SIZE,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": CATALOG_MARKET_FILTER,
        "fields": "f12,f14",
    }


def _fetch_catalog_page(request_get, page_number):
    """读取并校验目录接口的单页响应，不在这里修改本地文件。"""
    response = request_get(
        EASTMONEY_CATALOG_URL,
        params=_catalog_request_params(page_number),
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json().get("data") or {}
    total = payload.get("total")
    rows = payload.get("diff")
    if not isinstance(total, int) or total <= 0 or not isinstance(rows, list):
        raise ValueError("A 股代码目录接口返回的总数或列表格式异常")
    return total, rows


def refresh_catalog(
    request_get=requests.get,
    catalog_file=CATALOG_FILE,
    minimum_entries=MIN_A_SHARE_CATALOG_SIZE,
):
    """分页刷新完整 A 股名称目录；数量异常时保留旧目录不覆盖。"""
    try:
        total, rows = _fetch_catalog_page(request_get, 1)
        differences = list(rows)
        page_count = (total + CATALOG_PAGE_SIZE - 1) // CATALOG_PAGE_SIZE
        for page_number in range(2, page_count + 1):
            page_total, page_rows = _fetch_catalog_page(request_get, page_number)
            if page_total != total:
                raise ValueError("A 股代码目录分页返回的总数不一致")
            differences.extend(page_rows)

        entries = [
            {"code": item.get("f12", ""), "name": item.get("f14", "")}
            for item in differences
            if isinstance(item, dict)
        ]
        normalized_entries = normalize_catalog(entries)
        if len(normalized_entries) < minimum_entries:
            raise ValueError(
                f"仅获得 {len(normalized_entries)} 只有效沪深 A 股，低于完整目录校验阈值 {minimum_entries}"
            )
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        return {"status": "failed", "message": f"更新 A 股代码目录失败：{error}。"}

    catalog = merge_catalog_entries(normalized_entries, catalog_file)
    catalog_data = read_json(catalog_file, "A 股代码目录")
    catalog_data["目录来源"] = EASTMONEY_CATALOG_SOURCE
    write_json_atomically(catalog_file, catalog_data)
    return {
        "status": "success",
        "message": "A 股代码目录更新成功。",
        "count": len(catalog),
        "source": EASTMONEY_CATALOG_SOURCE,
    }


def resolve_code_query(query):
    """仅解析六位沪深 A 股代码；此路径不读取或依赖名称目录。"""
    clean_query = str(query).strip()
    if not (clean_query.isdigit() and len(clean_query) == 6):
        return None
    try:
        return {"code": normalize_stock_code(clean_query), "name": clean_query}
    except ValueError:
        return None


def resolve_catalog_query(query, catalog):
    """在本地目录按六位代码、全名或模糊名称查询，不请求网络。"""
    clean_query = str(query).strip()
    if not clean_query:
        return []
    if resolve_code_query(clean_query):
        return []
    exact_matches = [stock for stock in catalog if stock["name"] == clean_query]
    if exact_matches:
        return exact_matches
    return [stock for stock in catalog if clean_query in stock["name"]]


def fetch_stock_identity(stock_code, request_get=requests.get):
    """在用户明确下载时从腾讯行情接口确认股票名称。"""
    code = normalize_stock_code(stock_code)
    market_code = create_market_code(code)
    try:
        response = request_get(
            f"https://qt.gtimg.cn/q={market_code}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        parts = response.text.split('"', 2)[1].split("~")
        name = parts[1].strip()
    except (requests.RequestException, IndexError, AttributeError) as error:
        raise ValueError(f"无法识别股票代码 {code}：{error}。") from error
    if not name:
        raise ValueError(f"未识别到股票代码 {code} 对应的 A 股名称。")
    return {"code": code, "name": name}


def create_stock_record(stock, history_data, current_score, previous_score=None):
    """创建 stock_analysis 可直接复用的事实记录与日间变化事实。"""
    history = history_data.copy()
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    history["收盘"] = pd.to_numeric(history["收盘"], errors="coerce")
    history = history.dropna(subset=["日期", "收盘"]).sort_values("日期")
    if history.empty:
        raise ValueError("历史日线缺少有效日期或收盘价。")
    latest = history.iloc[-1]
    current_quant = {
        "股票代码": stock["code"],
        "股票名称": stock["name"],
        "综合评分": current_score["评分"],
        "RSI": current_score["RSI"],
        "MA5": current_score["MA5"],
        "MA20": current_score["MA20"],
        "MACD": current_score["MACD"],
        "建议": current_score["建议"],
    }
    previous_quant = None
    if previous_score:
        previous_quant = {
            "股票代码": stock["code"],
            "股票名称": stock["name"],
            "综合评分": previous_score["评分"],
            "RSI": previous_score["RSI"],
            "MA5": previous_score["MA5"],
            "MA20": previous_score["MA20"],
            "MACD": previous_score["MACD"],
            "建议": previous_score["建议"],
        }
    signal_stock = create_signal_stock(
        {"code": stock["code"], "name": stock["name"]},
        current_quant,
        previous_quant,
        None,
    )
    current = signal_stock["当前指标"]
    stock_record = {
        "code": stock["code"],
        "name": stock["name"],
        "alias": "",
        "priority": None,
        "tags": [],
        "notes": "",
        "cost_price": None,
        "target_price": None,
        "score": current["Score"],
        "advice": current["建议"],
        "trend": current["趋势"],
        "rsi": current["RSI"],
        "ma5": current["MA5"],
        "ma20": current["MA20"],
        "macd": current["MACD"],
        "risk": current["风险标签"],
        "source": "on_demand",
    }
    return {
        "股票代码": stock["code"],
        "股票名称": stock["name"],
        "数据日期": latest["日期"].strftime("%Y-%m-%d"),
        "日线数据说明": "指标基于最近一个已下载交易日日线，不代表实时盘中行情。",
        "stock_record": stock_record,
        "daily_signal": signal_stock,
    }


def save_history_atomically(history_data, data_file):
    """仅在已取得非空有效日线后原子保存 CSV。"""
    if history_data.empty:
        raise ValueError("行情接口返回的数据为空，未保存 CSV 文件。")
    data_file = Path(data_file)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = data_file.with_suffix(data_file.suffix + ".tmp")
    history_data.to_csv(temporary_file, index=False, encoding="utf-8-sig")
    os.replace(temporary_file, data_file)
    return data_file


def analyze_history(stock, history_file):
    """使用同一 score.py 评分函数分析本地日线，并比较上一根日线。"""
    history_data = pd.read_csv(history_file)
    if len(history_data) < 27:
        raise ValueError("历史日线不足，无法按现有 MA20 与 MACD 评分规则分析。")
    current_score = calculate_score_dataframe(history_data)
    try:
        previous_score = calculate_score_dataframe(history_data.iloc[:-1])
    except (IndexError, KeyError, ValueError):
        previous_score = None
    return create_stock_record(stock, history_data, current_score, previous_score)


def load_on_demand_snapshot(stock_code, output_directory=ON_DEMAND_OUTPUT_DIRECTORY):
    """读取隔离的临时分析快照；不存在时返回 None。"""
    snapshot_file = Path(output_directory) / f"on_demand_{normalize_stock_code(stock_code)}.json"
    if not snapshot_file.is_file():
        return None
    return read_json(snapshot_file, "按需分析快照")


def analyze_on_demand_stock(
    stock,
    refresh=False,
    get_history=get_stock,
    identity_fetcher=fetch_stock_identity,
    data_directory=ON_DEMAND_DATA_DIRECTORY,
    output_directory=ON_DEMAND_OUTPUT_DIRECTORY,
    catalog_file=CATALOG_FILE,
):
    """按需下载一只股票并分析；失败时保留原 CSV、快照且返回中文错误。"""
    try:
        code = normalize_stock_code(stock.get("code", ""))
    except (AttributeError, ValueError) as error:
        return {"status": "failed", "message": f"股票代码无效：{error}"}

    stock = {
        "code": code,
        "name": str(stock.get("name", "")).strip() or code,
        "名称目录来源": stock.get("名称目录来源", LOCAL_CATALOG_SOURCE),
    }
    previous_snapshot = load_on_demand_snapshot(code, output_directory)
    if previous_snapshot and not refresh:
        return {"status": "cached", "message": "已复用本地按需分析缓存。", "analysis": previous_snapshot}

    if stock["name"] == code:
        try:
            stock = identity_fetcher(code)
            stock["名称目录来源"] = QUOTE_SOURCE
        except ValueError as error:
            return {"status": "failed", "message": str(error)}

    history_file = get_history_file(stock, data_directory)
    try:
        raw_data = get_history(create_market_code(code))
        history_data = create_history_dataframe(raw_data)
        save_history_atomically(history_data, history_file)
        analysis = analyze_history(stock, history_file)
    except Exception as error:
        return {
            "status": "failed",
            "message": f"下载或分析 {stock['name']} 失败：{error}。已有本地缓存未被删除。",
        }

    try:
        data_file_display = str(history_file.relative_to(PROJECT_DIRECTORY))
    except ValueError:
        data_file_display = str(history_file)
    analysis.update(
        {
            "数据文件": data_file_display,
            "刷新时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "行情来源": MARKET_DATA_SOURCE,
            "名称目录来源": stock["名称目录来源"],
        }
    )
    snapshot_file = get_snapshot_file(code, output_directory)
    write_json_atomically(snapshot_file, analysis)
    merge_catalog_entries([stock], catalog_file)
    return {"status": "success", "message": "单股日线下载并分析成功。", "analysis": analysis}


def add_stock_to_watchlist(stock, watchlist_file=None):
    """仅在用户明确调用时将股票写入 watchlist.json，供下一次股票池更新使用。"""
    watchlist_path = Path(watchlist_file or PROJECT_DIRECTORY / "watchlist.json")
    try:
        watchlist_data = read_json(watchlist_path, "watchlist.json")
    except ValueError as error:
        return {"status": "failed", "message": str(error)}
    stocks = watchlist_data.get("stocks") if isinstance(watchlist_data, dict) else None
    if not isinstance(stocks, list):
        return {"status": "failed", "message": "watchlist.json 中的 stocks 必须是列表。"}
    try:
        code = normalize_stock_code(stock.get("code", ""))
    except (AttributeError, ValueError) as error:
        return {"status": "failed", "message": f"股票代码无效：{error}"}
    if any(isinstance(item, dict) and str(item.get("code", "")).zfill(6) == code for item in stocks):
        return {"status": "exists", "message": "该股票已在关注列表中。"}

    stocks.append(
        {
            "code": code,
            "name": str(stock.get("name", "")).strip() or code,
            "alias": "",
            "priority": 0,
            "enable": True,
            "tags": [],
            "cost_price": None,
            "target_price": None,
            "notes": "",
        }
    )
    write_json_atomically(watchlist_path, watchlist_data)
    return {"status": "success", "message": "已加入关注列表；将在下一次股票池更新时生效。"}
