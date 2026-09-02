"""AStockAI 统一股票池：合并原有股票池与已启用关注股票。"""

import json
from pathlib import Path

from astock_core.data.stock_pool import stocks as BASE_STOCKS


def normalize_stock_code(stock_code):
    """校验并规范化沪深 A 股六位股票代码。"""
    clean_code = str(stock_code).strip().zfill(6)

    if not clean_code.isdigit() or len(clean_code) != 6:
        raise ValueError(f"股票代码格式错误：{stock_code}。")

    if clean_code[0] not in {"0", "3", "6"}:
        raise ValueError(f"暂不支持该市场的股票代码：{clean_code}。")

    return clean_code


def create_market_code(stock_code):
    """根据六位股票代码生成既有行情接口使用的市场代码。"""
    normalized_code = normalize_stock_code(stock_code)
    market_prefix = "sh" if normalized_code.startswith("6") else "sz"
    return f"{market_prefix}{normalized_code}"


def load_base_stock_universe():
    """从项目原有 stock_pool.py 读取基础股票池。"""
    base_universe = []

    for stock_name, market_code in BASE_STOCKS.items():
        stock_code = normalize_stock_code(market_code[2:])
        base_universe.append(
            {
                "code": stock_code,
                "name": stock_name,
                "market_code": market_code,
                "source": "base",
            }
        )

    return base_universe


def load_watchlist_json(watchlist_file):
    """读取 watchlist.json，并在缺失或格式错误时给出明确提示。"""
    if not watchlist_file.exists():
        raise FileNotFoundError(f"未找到 watchlist.json：{watchlist_file}。")

    try:
        with open(watchlist_file, "r", encoding="utf-8") as file:
            watchlist_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"watchlist.json JSON 格式错误：{error.msg}。") from error
    except OSError as error:
        raise ValueError(f"无法读取 watchlist.json：{error}。") from error

    if not isinstance(watchlist_data, dict):
        raise ValueError("watchlist.json 顶层必须是对象。")

    return watchlist_data


def find_base_stock_by_code(base_universe, stock_code):
    """根据股票代码在基础股票池中查找已有股票名称。"""
    for stock in base_universe:
        if stock["code"] == stock_code:
            return stock

    return None


def normalize_watchlist_stock(stock_item, base_universe):
    """将新旧格式的关注股票统一为股票池记录。"""
    if isinstance(stock_item, str):
        stock_code = normalize_stock_code(stock_item)
        base_stock = find_base_stock_by_code(base_universe, stock_code)
        stock_name = base_stock["name"] if base_stock else stock_code
        enabled = True
    elif isinstance(stock_item, dict):
        stock_code = normalize_stock_code(stock_item.get("code", ""))
        base_stock = find_base_stock_by_code(base_universe, stock_code)
        stock_name = str(stock_item.get("name", "")).strip()
        stock_name = stock_name or (base_stock["name"] if base_stock else stock_code)
        enabled = stock_item.get("enable", True) is not False
    else:
        raise ValueError("watchlist.json 的 stocks 中包含无效股票项目。")

    return {
        "code": stock_code,
        "name": stock_name,
        "market_code": create_market_code(stock_code),
        "enable": enabled,
        "source": "watchlist",
    }


def load_watchlist_stocks(watchlist_file=None):
    """读取 watchlist.json 中所有股票，并兼容旧版字符串列表。"""
    project_directory = Path(__file__).parents[2]
    watchlist_path = watchlist_file or project_directory / "watchlist.json"
    watchlist_data = load_watchlist_json(watchlist_path)
    stock_items = watchlist_data.get("stocks", [])

    if not isinstance(stock_items, list):
        raise ValueError("watchlist.json 中的 stocks 必须是列表。")

    base_universe = load_base_stock_universe()
    return [
        normalize_watchlist_stock(stock_item, base_universe)
        for stock_item in stock_items
    ]


def merge_stock_universe(base_universe, watchlist_stocks, research_stocks=None):
    """按代码合并基础、研究与关注池；关注列表拥有最高展示优先级。"""
    merged_stocks = []
    stock_index = {}

    for stock in base_universe:
        stock_index[stock["code"]] = len(merged_stocks)
        merged_stocks.append(dict(stock))

    for stock in research_stocks or []:
        existing_index = stock_index.get(stock["code"])
        if existing_index is None:
            stock_index[stock["code"]] = len(merged_stocks)
            merged_stocks.append(dict(stock))
        else:
            merged_stocks[existing_index].update({
                "name": stock["name"], "market_code": stock["market_code"], "source": "base+research:csi300",
            })

    for stock in watchlist_stocks:
        if not stock["enable"]:
            continue

        existing_index = stock_index.get(stock["code"])
        if existing_index is None:
            stock_index[stock["code"]] = len(merged_stocks)
            merged_stocks.append(dict(stock))
            continue

        merged_stocks[existing_index].update(
            {
                "name": stock["name"],
                "market_code": stock["market_code"],
                "source": "base+watchlist",
            }
        )

    return merged_stocks


def get_enabled_stock_universe(watchlist_file=None):
    """返回基础股票池与已启用关注股票合并后的去重股票池。"""
    base_universe = load_base_stock_universe()
    watchlist_stocks = load_watchlist_stocks(watchlist_file)
    # 延迟导入避免研究池模块反向依赖股票池工具。
    from astock_core.research.research_universe import load_research_universe
    research_stocks = load_research_universe()
    return merge_stock_universe(base_universe, watchlist_stocks, research_stocks)


def create_stock_code_lookup(stock_universe):
    """建立股票名称到六位股票代码的查询字典。"""
    return {stock["name"]: stock["code"] for stock in stock_universe}
