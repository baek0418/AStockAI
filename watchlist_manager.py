"""AStockAI 关注股票配置的读取、校验和安全写入模块。"""

import copy
import json
import os
import shutil
from pathlib import Path

from stock_universe import load_base_stock_universe


def get_watchlist_file(watchlist_file=None):
    """返回指定或项目默认的 watchlist.json 路径。"""
    if watchlist_file:
        return Path(watchlist_file)

    return Path(__file__).parent / "watchlist.json"


def load_watchlist_data(watchlist_file=None):
    """读取 watchlist.json 原始数据，并检查基础结构。"""
    file_path = get_watchlist_file(watchlist_file)

    if not file_path.exists():
        raise FileNotFoundError(f"未找到关注列表文件：{file_path}。")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            watchlist_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"watchlist.json JSON 格式错误：{error.msg}。") from error
    except OSError as error:
        raise ValueError(f"无法读取 watchlist.json：{error}。") from error

    validate_watchlist_data(watchlist_data, allow_empty=True)
    return watchlist_data


def validate_stock_code(stock_code):
    """校验股票代码是否为六位数字字符串。"""
    clean_code = str(stock_code).strip()

    if len(clean_code) != 6 or not clean_code.isdigit():
        raise ValueError("股票代码必须为 6 位数字。")

    return clean_code


def validate_stock_name(stock_name):
    """校验股票名称不能为空。"""
    clean_name = str(stock_name).strip()

    if not clean_name:
        raise ValueError("股票名称不能为空。")

    return clean_name


def validate_priority(priority):
    """校验优先级为 1 至 5 的整数。"""
    if isinstance(priority, bool):
        raise ValueError("优先级必须是 1 至 5 的整数。")

    try:
        clean_priority = int(priority)
    except (TypeError, ValueError) as error:
        raise ValueError("优先级必须是 1 至 5 的整数。") from error

    if clean_priority < 1 or clean_priority > 5:
        raise ValueError("优先级必须在 1 至 5 之间。")

    return clean_priority


def validate_optional_price(price, field_name):
    """校验成本价或目标价为正数或空值。"""
    if price is None or str(price).strip() == "":
        return None

    try:
        clean_price = float(price)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是正数或留空。") from error

    if clean_price <= 0:
        raise ValueError(f"{field_name}必须是正数或留空。")

    return clean_price


def parse_tags(tags_text):
    """将逗号分隔的标签文本转换为去空白的字符串列表。"""
    if isinstance(tags_text, list):
        source_tags = tags_text
    else:
        source_tags = str(tags_text).split(",")

    return [str(tag).strip() for tag in source_tags if str(tag).strip()]


def get_base_name_by_code(stock_code):
    """从原有基础股票池中查找旧版代码对应的名称。"""
    for stock in load_base_stock_universe():
        if stock["code"] == stock_code:
            return stock["name"]

    return "未知股票"


def get_display_stock(stock_item, index):
    """将新旧格式配置转换为可展示和编辑的统一记录。"""
    if isinstance(stock_item, str):
        stock_code = validate_stock_code(stock_item)
        return {
            "index": index,
            "code": stock_code,
            "name": get_base_name_by_code(stock_code),
            "alias": "",
            "priority": 3,
            "enable": True,
            "tags": [],
            "cost_price": None,
            "target_price": None,
            "notes": "",
            "legacy": True,
            "raw": stock_item,
        }

    if not isinstance(stock_item, dict):
        raise ValueError("关注列表中包含非字符串、非对象的无效项目。")

    stock_code = validate_stock_code(stock_item.get("code", ""))
    return {
        "index": index,
        "code": stock_code,
        "name": validate_stock_name(stock_item.get("name", "")),
        "alias": str(stock_item.get("alias", "")).strip(),
        "priority": validate_priority(stock_item.get("priority", 3)),
        "enable": stock_item.get("enable", True) is not False,
        "tags": parse_tags(stock_item.get("tags", [])),
        "cost_price": validate_optional_price(stock_item.get("cost_price"), "持仓成本"),
        "target_price": validate_optional_price(stock_item.get("target_price"), "目标价"),
        "notes": str(stock_item.get("notes", "")).strip(),
        "legacy": False,
        "raw": stock_item,
    }


def get_watchlist_entries(watchlist_data):
    """按原始顺序返回关注列表的统一展示记录。"""
    return [
        get_display_stock(stock_item, index)
        for index, stock_item in enumerate(watchlist_data["stocks"], start=1)
    ]


def validate_watchlist_data(watchlist_data, allow_empty=False):
    """校验关注列表结构、股票字段和代码唯一性。"""
    if not isinstance(watchlist_data, dict):
        raise ValueError("watchlist.json 顶层必须是对象。")

    stock_items = watchlist_data.get("stocks")
    if not isinstance(stock_items, list):
        raise ValueError("watchlist.json 中的 stocks 必须是列表。")

    if not allow_empty and not stock_items:
        raise ValueError("关注列表不能为空，无法覆盖原配置。")

    stock_codes = set()
    for entry in get_watchlist_entries(watchlist_data):
        if entry["code"] in stock_codes:
            raise ValueError(f"股票代码重复：{entry['code']}。")
        stock_codes.add(entry["code"])

    return True


def create_stock_config(
    code,
    name,
    alias="",
    priority=3,
    tags=None,
    cost_price=None,
    target_price=None,
    notes="",
):
    """根据用户输入创建经过校验的新版关注股票配置。"""
    return {
        "code": validate_stock_code(code),
        "name": validate_stock_name(name),
        "alias": str(alias).strip(),
        "priority": validate_priority(priority),
        "enable": True,
        "tags": parse_tags(tags or []),
        "cost_price": validate_optional_price(cost_price, "持仓成本"),
        "target_price": validate_optional_price(target_price, "目标价"),
        "notes": str(notes).strip(),
    }


def find_watchlist_entry(watchlist_data, query):
    """按序号、代码、全名或唯一模糊名称和别名定位关注股票。"""
    entries = get_watchlist_entries(watchlist_data)
    clean_query = str(query).strip()

    if not clean_query:
        return None, "请输入序号、股票代码或名称。", []

    code_matches = [entry for entry in entries if entry["code"] == clean_query]
    if len(code_matches) == 1:
        return code_matches[0], None, []

    name_matches = [entry for entry in entries if entry["name"] == clean_query]
    if len(name_matches) == 1:
        return name_matches[0], None, []

    if clean_query.isdigit() and len(clean_query) < 6:
        selected_index = int(clean_query)
        for entry in entries:
            if entry["index"] == selected_index:
                return entry, None, []

    fuzzy_matches = [
        entry
        for entry in entries
        if clean_query in entry["name"] or clean_query in entry["alias"]
    ]

    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], None, []

    if len(fuzzy_matches) > 1:
        return None, "找到多个关注股票，请输入完整名称、代码或序号。", fuzzy_matches

    return None, "未找到关注股票。", []


def make_editable_stock(entry):
    """将旧版或新版展示记录转换为可安全保存的新版配置。"""
    if entry["legacy"]:
        return create_stock_config(
            entry["code"],
            entry["name"],
            entry["alias"],
            entry["priority"],
            entry["tags"],
            entry["cost_price"],
            entry["target_price"],
            entry["notes"],
        )

    return copy.deepcopy(entry["raw"])


def replace_stock(watchlist_data, entry_index, stock_config):
    """替换指定序号的配置，并保留其他股票原始顺序。"""
    updated_data = copy.deepcopy(watchlist_data)
    updated_data["stocks"][entry_index - 1] = stock_config
    validate_watchlist_data(updated_data)
    return updated_data


def append_stock(watchlist_data, stock_config):
    """向关注列表末尾添加经过校验的新股票配置。"""
    updated_data = copy.deepcopy(watchlist_data)
    updated_data["stocks"].append(stock_config)
    validate_watchlist_data(updated_data)
    return updated_data


def remove_stock(watchlist_data, entry_index):
    """删除指定序号的关注股票配置，不处理任何历史行情文件。"""
    updated_data = copy.deepcopy(watchlist_data)
    del updated_data["stocks"][entry_index - 1]
    validate_watchlist_data(updated_data)
    return updated_data


def save_watchlist_data(watchlist_data, watchlist_file=None):
    """通过备份、临时文件和原子替换安全保存关注列表。"""
    validate_watchlist_data(watchlist_data)
    file_path = get_watchlist_file(watchlist_file)
    temporary_file = file_path.with_name(f"{file_path.name}.tmp")
    backup_file = file_path.with_name(f"{file_path.name}.bak")

    try:
        if file_path.exists():
            shutil.copy2(file_path, backup_file)

        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(watchlist_data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_file, file_path)
        saved_data = load_watchlist_data(file_path)
        validate_watchlist_data(saved_data)
    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise

    return backup_file


def get_watchlist_statistics(watchlist_data):
    """统计关注列表总数、启用数和停用数。"""
    entries = get_watchlist_entries(watchlist_data)
    enabled_count = sum(entry["enable"] for entry in entries)

    return {
        "total": len(entries),
        "enabled": enabled_count,
        "disabled": len(entries) - enabled_count,
    }
