"""AStockAI v3.1：根据关注股票配置生成量化事实快照。"""

import json
from pathlib import Path


def parse_json_content(json_content, file_description):
    """解析 JSON 文本，并在格式错误时返回明确提示。"""
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as error:
        raise ValueError(f"{file_description} JSON 格式错误：{error.msg}。") from error


def load_json_file(file_path, file_description):
    """读取指定 JSON 文件，并检查文件是否存在。"""
    if not file_path.exists():
        raise FileNotFoundError(f"未找到{file_description}：{file_path}。")

    try:
        json_content = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"无法读取{file_description}：{error}。") from error

    return parse_json_content(json_content, file_description)


def create_legacy_watchlist_stock(stock_code):
    """将旧版字符串股票代码转换为基础关注股票配置。"""
    return {
        "code": str(stock_code).strip(),
        "name": "未知股票",
        "alias": "",
        "priority": 0,
        "enable": True,
        "tags": [],
        "cost_price": None,
        "target_price": None,
        "notes": "",
    }


def normalize_watchlist_stock(stock_item):
    """整理单只关注股票配置，并忽略未使用的未来字段。"""
    if isinstance(stock_item, str):
        return create_legacy_watchlist_stock(stock_item)

    if not isinstance(stock_item, dict):
        raise ValueError("watchlist.json 的 stocks 中包含非字符串、非对象的项目。")

    stock_code = str(stock_item.get("code", "")).strip()
    stock_name = str(stock_item.get("name", "")).strip()

    if not stock_code:
        raise ValueError("watchlist.json 中存在缺少 code 的股票配置。")

    if not stock_name:
        raise ValueError(f"watchlist.json 中股票 {stock_code} 缺少 name。")

    tags = stock_item.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    return {
        "code": stock_code,
        "name": stock_name,
        "alias": str(stock_item.get("alias", "")).strip(),
        "priority": stock_item.get("priority", 0),
        "enable": stock_item.get("enable", True),
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        "cost_price": stock_item.get("cost_price"),
        "target_price": stock_item.get("target_price"),
        "notes": str(stock_item.get("notes", "")).strip(),
    }


def create_watchlist_configs(stock_items):
    """校验并整理关注股票列表，且不改变用户原始顺序。"""
    if not isinstance(stock_items, list):
        raise ValueError("watchlist.json 中的 stocks 必须是列表。")

    if not stock_items:
        raise ValueError("watchlist.json 的 stocks 为空，无法生成关注股票快照。")

    return [normalize_watchlist_stock(stock_item) for stock_item in stock_items]


def load_watchlist(watchlist_file):
    """读取 watchlist.json，并保留用户原始配置顺序。"""
    watchlist_data = load_json_file(watchlist_file, "watchlist.json")

    if not isinstance(watchlist_data, dict):
        raise ValueError("watchlist.json 顶层必须是对象。")

    return create_watchlist_configs(watchlist_data.get("stocks"))


def find_quant_snapshot_file(output_directory):
    """查找 output 文件夹中最新的 quant_snapshot JSON 文件。"""
    fixed_snapshot_file = output_directory / "quant_snapshot.json"
    if fixed_snapshot_file.exists():
        return fixed_snapshot_file

    snapshot_files = sorted(output_directory.glob("quant_snapshot_*.json"))
    if not snapshot_files:
        raise FileNotFoundError("output 文件夹中没有 quant_snapshot JSON 文件。")

    return snapshot_files[-1]


def load_quant_snapshot(snapshot_file):
    """读取量化事实快照，并检查股票排行榜字段。"""
    quant_snapshot = load_json_file(snapshot_file, "quant_snapshot.json")

    if not isinstance(quant_snapshot, dict):
        raise ValueError("quant_snapshot.json 顶层必须是对象。")

    if not quant_snapshot.get("快照日期"):
        raise ValueError("quant_snapshot.json 缺少快照日期。")

    stock_rankings = quant_snapshot.get("股票排行榜")
    if not isinstance(stock_rankings, list):
        raise ValueError("quant_snapshot.json 的股票排行榜必须是列表。")

    return quant_snapshot


def get_snapshot_stock_code(stock_data):
    """从量化快照股票记录中读取可能存在的股票代码字段。"""
    for field_name in ("股票代码", "代码", "code"):
        stock_code = stock_data.get(field_name)
        if stock_code is not None and str(stock_code).strip():
            return str(stock_code).strip()

    return ""


def create_snapshot_lookups(stock_rankings):
    """按股票代码和股票名称建立量化快照查询索引。"""
    code_lookup = {}
    name_lookup = {}

    for stock_data in stock_rankings:
        if not isinstance(stock_data, dict):
            continue

        stock_code = get_snapshot_stock_code(stock_data)
        stock_name = str(stock_data.get("股票名称", "")).strip()

        if stock_code:
            code_lookup[stock_code] = stock_data

        if stock_name:
            name_lookup[stock_name] = stock_data

    return code_lookup, name_lookup


def find_matching_stock(watchlist_stock, code_lookup, name_lookup):
    """优先按代码、其次按名称查找对应的量化快照股票记录。"""
    stock_data = code_lookup.get(watchlist_stock["code"])

    if stock_data is not None:
        return stock_data

    return name_lookup.get(watchlist_stock["name"])


def create_trend_label(stock_data):
    """基于量化快照中已有的 MA5 和 MA20 生成均线趋势标签。"""
    ma5 = stock_data.get("MA5")
    ma20 = stock_data.get("MA20")

    if not isinstance(ma5, (int, float)) or not isinstance(ma20, (int, float)):
        return "快照未提供"

    if ma5 > ma20:
        return "均线多头"

    if ma5 < ma20:
        return "均线偏弱"

    return "均线持平"


def create_risk_label(stock_data):
    """根据量化快照中已有的建议字段整理风险状态。"""
    advice = stock_data.get("建议")

    if advice == "风险":
        return "风险"

    if advice is None:
        return "快照未提供"

    return "正常"


def create_matched_stock(watchlist_stock, stock_data):
    """组合关注配置和已有量化事实，创建已匹配股票记录。"""
    return {
        "code": watchlist_stock["code"],
        "name": watchlist_stock["name"],
        "alias": watchlist_stock["alias"],
        "priority": watchlist_stock["priority"],
        "enable": watchlist_stock["enable"],
        "tags": watchlist_stock["tags"],
        "score": stock_data.get("综合评分"),
        "advice": stock_data.get("建议"),
        "trend": create_trend_label(stock_data),
        "rsi": stock_data.get("RSI"),
        "ma5": stock_data.get("MA5"),
        "ma20": stock_data.get("MA20"),
        "macd": stock_data.get("MACD"),
        "risk": create_risk_label(stock_data),
    }


def create_missing_stock(watchlist_stock):
    """创建未在量化快照中找到的关注股票记录。"""
    return {
        "code": watchlist_stock["code"],
        "name": watchlist_stock["name"],
        "status": "missing",
    }


def create_watchlist_snapshot(watchlist_stocks, quant_snapshot):
    """按关注列表原始顺序生成完整的关注股票量化快照。"""
    stock_rankings = quant_snapshot["股票排行榜"]
    code_lookup, name_lookup = create_snapshot_lookups(stock_rankings)
    snapshot_stocks = []
    matched_count = 0

    for watchlist_stock in watchlist_stocks:
        stock_data = find_matching_stock(watchlist_stock, code_lookup, name_lookup)

        if stock_data is None:
            snapshot_stocks.append(create_missing_stock(watchlist_stock))
            continue

        snapshot_stocks.append(create_matched_stock(watchlist_stock, stock_data))
        matched_count += 1

    return {
        "date": quant_snapshot["快照日期"],
        "total_watchlist": len(watchlist_stocks),
        "matched": matched_count,
        "missing": len(watchlist_stocks) - matched_count,
        "stocks": snapshot_stocks,
    }


def save_watchlist_snapshot(watchlist_snapshot, output_directory):
    """将关注股票量化快照保存到 output 文件夹。"""
    output_directory.mkdir(exist_ok=True)
    snapshot_file = output_directory / (
        f"watchlist_snapshot_{watchlist_snapshot['date']}.json"
    )

    with open(snapshot_file, "w", encoding="utf-8") as file:
        json.dump(watchlist_snapshot, file, ensure_ascii=False, indent=2)

    return snapshot_file


def run_watchlist_snapshot():
    """执行关注股票量化快照生成流程，并在异常时输出明确提示。"""
    project_directory = Path(__file__).parent

    try:
        watchlist_stocks = load_watchlist(project_directory / "watchlist.json")
        snapshot_file = find_quant_snapshot_file(project_directory / "output")
        quant_snapshot = load_quant_snapshot(snapshot_file)
        watchlist_snapshot = create_watchlist_snapshot(
            watchlist_stocks,
            quant_snapshot,
        )
        output_file = save_watchlist_snapshot(
            watchlist_snapshot,
            project_directory / "output",
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"关注股票快照生成失败：{error}")
        return None

    print("关注股票快照生成成功:")
    print(output_file)
    return watchlist_snapshot, output_file


if __name__ == "__main__":
    run_watchlist_snapshot()
