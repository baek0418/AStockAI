"""更新统一股票池中的真实历史行情数据。"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from market_data_sources import MarketDataFetchResult, fetch_daily_history
from process_journal import ProcessJournal
from stock_universe import get_enabled_stock_universe


def get_stock(code):
    """从腾讯行情接口获取个股日线，优先使用前复权数据。"""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{code},day,,,600,qfq"}
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    response_data = response.json()

    try:
        payload = response_data["data"][code]
    except (KeyError, TypeError) as error:
        raise ValueError(f"行情接口未返回 {code} 的历史数据。") from error

    # 少数股票（例如国机重装）不提供 qfqday，但同一响应中仍有完整的
    # 未复权 day 日线。此时宁可明确回退并保留可用历史，也不把它误报为
    # "数据不足"；优先级仍始终是前复权数据。
    adjusted_rows = payload.get("qfqday") if isinstance(payload, dict) else None
    if isinstance(adjusted_rows, list) and adjusted_rows:
        return adjusted_rows

    raw_rows = payload.get("day") if isinstance(payload, dict) else None
    if isinstance(raw_rows, list) and raw_rows:
        return raw_rows

    raise ValueError(f"行情接口未返回 {code} 的有效 qfqday 或 day 历史数据。")


def create_history_dataframe(raw_data):
    """将接口返回的真实日线数据转换为项目统一的 DataFrame。"""
    rows = []

    for item in raw_data:
        rows.append(
            [
                item[0],
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
            ]
        )

    history_data = pd.DataFrame(
        rows,
        columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"],
    )

    if history_data.empty:
        raise ValueError("行情接口返回的数据为空，未保存 CSV 文件。")

    return history_data


def _write_provenance(data_directory, stock_code, market_code, result):
    """保存行情来源审计信息；与正式 CSV 分目录存放，绝不被研究扫描读取。"""
    provenance_directory = Path(data_directory) / "provenance"
    provenance_directory.mkdir(parents=True, exist_ok=True)
    provenance_file = provenance_directory / f"{stock_code}.json"
    temporary_file = provenance_file.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(result.provenance(stock_code, market_code), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(provenance_file)
    return provenance_file


def _save_stock_record(stock, data_directory, fetch_history=fetch_daily_history):
    """更新一只股票，并返回主备来源、文件和失败信息均明确的记录。"""
    name = stock["name"]
    market_code = stock["market_code"]
    stock_code = stock["code"]
    result = fetch_history(market_code)
    if not isinstance(result, MarketDataFetchResult):
        raise ValueError("行情数据源未返回可审计的日线结果。")
    history_data = result.history
    data_file = Path(data_directory) / f"{name}历史.csv"
    temporary_file = data_file.with_suffix(".csv.tmp")
    history_data.to_csv(temporary_file, index=False, encoding="utf-8-sig")
    temporary_file.replace(data_file)
    provenance_file = _write_provenance(data_directory, stock_code, market_code, result)
    return {
        "code": stock_code,
        "name": name,
        "output_file": str(data_file),
        "provenance_file": str(provenance_file),
        "source": result.source,
        "adjustment": result.adjustment,
        "used_fallback": result.used_fallback,
        "rows": int(len(history_data)),
    }


def save_stock(name, code, data_directory, fetch_history=fetch_daily_history):
    """兼容旧接口：更新单只股票并返回正式 CSV 路径。"""
    stock_code = str(code)[2:] if str(code).startswith(("sh", "sz")) else str(code)
    record = _save_stock_record(
        {"name": name, "code": stock_code, "market_code": code}, data_directory, fetch_history
    )
    return Path(record["output_file"])


def run_update_data(project_directory=None, stock_universe=None, max_workers=3, fetch_history=fetch_daily_history):
    """并发更新全股票池；每只股票主源失败后才整只切换备用源。"""
    project_directory = Path(project_directory or Path(__file__).parent)
    data_directory = project_directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    journal = ProcessJournal("market_data_update", project_directory)

    if stock_universe is None:
        try:
            stock_universe = get_enabled_stock_universe()
        except (FileNotFoundError, ValueError) as error:
            journal.event("初始化股票池", "failed", 原因=str(error))
            print("股票池读取失败:")
            print(error)
            return {
                "success": False,
                "status": "failed",
                "message": "股票池读取失败",
                "output_file": None,
                "details": {"success_count": 0, "failed_count": 0, "failed_stocks": []},
            }

    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers 必须是正整数。")
    stocks = list(stock_universe)
    journal.event("初始化股票池", "info", 股票数=len(stocks), 并发数=max_workers)
    if not stocks:
        journal.event("初始化股票池", "failed", 原因="股票池为空。")
        return {
            "success": False,
            "status": "failed",
            "message": "股票池为空",
            "output_file": None,
            "details": {"success_count": 0, "failed_count": 0, "failed_stocks": []},
        }

    success_stocks = []
    failed_stocks = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="行情更新") as executor:
        futures = {
            executor.submit(_save_stock_record, stock, data_directory, fetch_history): stock
            for stock in stocks
        }
        for future in as_completed(futures):
            stock = futures[future]
            try:
                record = future.result()
                success_stocks.append(record)
                journal.event(
                    "更新股票日线",
                    "partial" if record["used_fallback"] else "success",
                    股票代码=record["code"],
                    股票名称=record["name"],
                    数据源=record["source"],
                    复权方式=record["adjustment"],
                    有效行数=record["rows"],
                )
            except Exception as error:
                failed = {"code": stock.get("code", ""), "name": stock.get("name", ""), "error": str(error)}
                failed_stocks.append(failed)
                journal.event("更新股票日线", "failed", 股票代码=failed["code"], 股票名称=failed["name"], 原因=str(error))

    success_stocks.sort(key=lambda item: (item["code"], item["name"]))
    failed_stocks.sort(key=lambda item: (item["code"], item["name"]))

    if not failed_stocks:
        status = "success"
        message = "行情更新完成"
    elif success_stocks:
        status = "partial"
        message = "行情部分更新完成"
    else:
        status = "failed"
        message = "行情更新失败"

    journal.event(
        "全市场更新完成",
        status,
        成功股票数=len(success_stocks),
        失败股票数=len(failed_stocks),
        使用备用源股票数=sum(item["used_fallback"] for item in success_stocks),
    )

    return {
        "success": bool(success_stocks),
        "status": status,
        "message": message,
        "output_file": None,
        "details": {
            "success_count": len(success_stocks),
            "failed_count": len(failed_stocks),
            "fallback_count": sum(item["used_fallback"] for item in success_stocks),
            "success_stocks": success_stocks,
            "failed_stocks": failed_stocks,
            "event_log": str(journal.event_file),
        },
    }


if __name__ == "__main__":
    run_update_data()
