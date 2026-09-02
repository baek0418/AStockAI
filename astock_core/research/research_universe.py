"""版本化研究股票池：下载、保存并加载沪深300成分股快照。"""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import requests

from astock_core.data.stock_universe import create_market_code, normalize_stock_code


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()
CONFIG_FILE = PROJECT_DIRECTORY / "config" / "research_universe.json"
UNIVERSE_DIRECTORY = PROJECT_DIRECTORY / "config" / "universe"
EASTMONEY_CONSTITUENTS_URL = "https://push2.eastmoney.com/api/qt/clist/get"
CSI300_BOARD_CODE = "BK0500"
EXPECTED_CSI300_SIZE = 300
CONSTITUENTS_PAGE_SIZE = 100


def validate_csi300_stocks(stocks):
    """规范化并校验一份完整的沪深300成分股列表。"""
    if not isinstance(stocks, list) or len(stocks) != EXPECTED_CSI300_SIZE:
        raise ValueError("沪深300成分股数量不是 300，拒绝使用该列表。")
    normalized, seen = [], set()
    for stock in stocks:
        if not isinstance(stock, dict):
            raise ValueError("沪深300成分股包含无效记录。")
        code = normalize_stock_code(stock.get("code", ""))
        name = str(stock.get("name", "")).strip()
        if not name or code in seen:
            raise ValueError("沪深300成分股包含空名称或重复代码。")
        seen.add(code)
        normalized.append({"code": code, "name": name, "market_code": create_market_code(code)})
    return sorted(normalized, key=lambda stock: stock["code"])


def fetch_csi300_constituents(request_get=requests.get):
    """分页读取公开行情源的沪深300列表，并严格校验数量和代码。"""
    request_params = {
        "pz": CONSTITUENTS_PAGE_SIZE,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": f"b:{CSI300_BOARD_CODE}",
        "fields": "f12,f14",
    }

    def fetch_page(page_number):
        response = request_get(
            EASTMONEY_CONSTITUENTS_URL,
            params={"pn": page_number, **request_params},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json().get("data") or {}
        rows = payload.get("diff")
        if not isinstance(rows, list):
            raise ValueError("沪深300成分股接口未返回有效列表。")
        return payload.get("total"), rows

    try:
        total, rows = fetch_page(1)
        if total != EXPECTED_CSI300_SIZE:
            raise ValueError("沪深300成分股数量异常，拒绝覆盖已有研究股票池。")
        for page_number in range(2, math.ceil(total / CONSTITUENTS_PAGE_SIZE) + 1):
            _, page_rows = fetch_page(page_number)
            rows.extend(page_rows)
    except (requests.RequestException, ValueError) as error:
        raise ValueError("沪深300成分股下载失败，未改写研究股票池快照。") from error
    if len(rows) != EXPECTED_CSI300_SIZE:
        raise ValueError("沪深300成分股数量异常，拒绝覆盖已有研究股票池。")
    return validate_csi300_stocks([
        {"code": row.get("f12", ""), "name": row.get("f14", "")}
        for row in rows
    ])


def save_csi300_snapshot(
    stocks,
    universe_directory=UNIVERSE_DIRECTORY,
    now=None,
    source="Eastmoney BK0500 constituent endpoint",
    source_note="用于研究池构建；指数官方调样与历史成分以中证指数公告为准。",
    as_of_date=None,
):
    """原子保存一次成分股快照，保留来源与内容校验值。"""
    stocks = validate_csi300_stocks(stocks)
    now = now or datetime.now()
    as_of_date = as_of_date or now.strftime("%Y-%m-%d")
    serialized = json.dumps(stocks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    snapshot = {
        "universe": "csi300",
        "as_of_date": as_of_date,
        "downloaded_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "source_note": source_note,
        "stock_count": len(stocks),
        "content_sha256": hashlib.sha256(serialized).hexdigest(),
        "stocks": stocks,
    }
    directory = Path(universe_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"csi300_{snapshot['as_of_date']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def enable_snapshot(snapshot_path, config_file=CONFIG_FILE):
    """仅在完整快照已写入后启用研究池配置。"""
    config_path = Path(config_file)
    snapshot_path = Path(snapshot_path)
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"研究股票池快照读取失败：{error}。") from error
    validate_csi300_stocks(snapshot.get("stocks"))
    if snapshot.get("universe") != "csi300" or snapshot.get("stock_count") != EXPECTED_CSI300_SIZE:
        raise ValueError("研究股票池快照校验失败。")
    try:
        snapshot_name = str(snapshot_path.resolve().relative_to(config_path.parent.resolve()))
    except ValueError as error:
        raise ValueError("研究股票池快照必须保存于 config 目录内。") from error
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "version": 1,
        "enabled": True,
        "universe": "csi300",
        "snapshot_file": snapshot_name,
        "refresh_policy": "每月复核成分股；每日仅增量更新已启用股票的日线。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(config_path)


def load_research_universe(config_file=CONFIG_FILE):
    """加载启用的研究股票池；缺快照时安全返回空池。"""
    config_path = Path(config_file)
    if not config_path.is_file():
        return []
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"研究股票池配置读取失败：{error}。") from error
    if config.get("enabled") is not True:
        return []
    snapshot_name = config.get("snapshot_file")
    if not isinstance(snapshot_name, str) or not snapshot_name:
        raise ValueError("研究股票池已启用但未配置快照文件。")
    snapshot_path = config_path.parent / snapshot_name
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"研究股票池快照读取失败：{error}。") from error
    stocks = snapshot.get("stocks")
    if snapshot.get("universe") != "csi300" or snapshot.get("stock_count") != EXPECTED_CSI300_SIZE or not isinstance(stocks, list):
        raise ValueError("研究股票池快照校验失败。")
    if len(stocks) != EXPECTED_CSI300_SIZE:
        raise ValueError("研究股票池快照股票数不为 300。")
    normalized = []
    for stock in stocks:
        code = normalize_stock_code(stock.get("code", ""))
        name = str(stock.get("name", "")).strip()
        if not name:
            raise ValueError("研究股票池快照包含空名称。")
        normalized.append({"code": code, "name": name, "market_code": create_market_code(code), "source": "research:csi300"})
    if len({stock["code"] for stock in normalized}) != EXPECTED_CSI300_SIZE:
        raise ValueError("研究股票池快照包含重复代码。")
    return normalized


def refresh_csi300_snapshot(config_file=CONFIG_FILE):
    """下载并写入最新快照，同时更新启用配置。"""
    stocks = fetch_csi300_constituents()
    config_path = Path(config_file)
    snapshot = save_csi300_snapshot(stocks, config_path.parent / "universe")
    enable_snapshot(snapshot, config_path)
    return snapshot


def main():
    try:
        snapshot = refresh_csi300_snapshot()
    except ValueError as error:
        print(error)
        return 1
    print(f"沪深300研究股票池已更新：{snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
