"""仅由显式命令下载的腾讯指数原始日线；不复用个股 qfqday 解析。"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()
MARKET_DIRECTORY = PROJECT_DIRECTORY / "data" / "market"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
BENCHMARKS = {
    "沪深300": {"market_code": "sh000300", "file_name": "沪深300_sh000300.csv"},
    "中证1000": {"market_code": "sh000852", "file_name": "中证1000_sh000852.csv"},
}
REQUIRED_INDEX_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]


def extract_index_day_rows(response_data, market_code):
    """从腾讯响应读取指数 ``data[code].day``，不要求也不读取 qfqday。"""
    try:
        payload = response_data["data"][market_code]
    except (KeyError, TypeError) as error:
        raise ValueError(f"腾讯指数响应缺少 data[{market_code}]。") from error
    if not isinstance(payload, dict) or "day" not in payload:
        raise ValueError(f"腾讯指数 {market_code} 缺少 day 日线字段；指数不使用 qfqday。")
    rows = payload["day"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"腾讯指数 {market_code} 的 day 日线为空，未保存文件。")
    return rows


def fetch_tencent_index_daily(market_code, request_get=requests.get):
    """请求腾讯指数日线接口，返回未经变换的 ``day`` 行。"""
    response = request_get(
        TENCENT_KLINE_URL,
        params={"param": f"{market_code},day,,,600,qfq"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    return extract_index_day_rows(response.json(), market_code)


def normalize_index_dataframe(day_rows):
    """严格规范指数 day 行；任何日期或 OHLCV 字段异常都拒绝写入。"""
    normalized = []
    for row_number, row in enumerate(day_rows, start=1):
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise ValueError(f"指数 day 第 {row_number} 行字段不足，需包含日期、开高低收和成交量。")
        date = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(date):
            raise ValueError(f"指数 day 第 {row_number} 行日期无效。")
        try:
            open_price, close_price, high_price, low_price, volume = (float(value) for value in row[1:6])
        except (TypeError, ValueError) as error:
            raise ValueError(f"指数 day 第 {row_number} 行开高低收或成交量不是数值。") from error
        values = (open_price, close_price, high_price, low_price, volume)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"指数 day 第 {row_number} 行含非有限数值。")
        if min(open_price, close_price, high_price, low_price) <= 0 or volume < 0:
            raise ValueError(f"指数 day 第 {row_number} 行开高低收或成交量范围无效。")
        if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
            raise ValueError(f"指数 day 第 {row_number} 行最高/最低价关系无效。")
        normalized.append(
            [date.strftime("%Y-%m-%d"), open_price, close_price, high_price, low_price, volume]
        )
    frame = pd.DataFrame(normalized, columns=REQUIRED_INDEX_COLUMNS)
    frame = frame.drop_duplicates("日期", keep="last").sort_values("日期").reset_index(drop=True)
    if frame.empty:
        raise ValueError("腾讯指数日线为空，未保存文件。")
    return frame


def save_benchmark_atomically(history_data, output_file, metadata):
    """仅保存已通过严格验证的非空 CSV 及其来源/复权元数据。"""
    if history_data.empty:
        raise ValueError("腾讯接口未返回有效指数日线，未保存文件。")
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = target.with_suffix(target.suffix + ".tmp")
    metadata_file = target.with_suffix(".metadata.json")
    temporary_metadata = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    history_data.to_csv(temporary_csv, index=False, encoding="utf-8-sig")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(temporary_csv, target)
        os.replace(temporary_metadata, metadata_file)
    finally:
        temporary_csv.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return target, metadata_file


def download_benchmarks(selected_names=None, market_directory=MARKET_DIRECTORY, fetch_index_daily=fetch_tencent_index_daily):
    """显式下载两个必要指数；不下载股票、不写入 ``data/*.csv``。"""
    selected_names = selected_names or list(BENCHMARKS)
    results = []
    for name in selected_names:
        definition = BENCHMARKS.get(name)
        if not definition:
            results.append({"名称": name, "status": "failed", "message": "未知基准名称。"})
            continue
        try:
            day_rows = fetch_index_daily(definition["market_code"])
            history_data = normalize_index_dataframe(day_rows)
            metadata = {
                "name": name,
                "market_code": definition["market_code"],
                "data_type": "index_daily",
                "adjustment": "none",
                "source": "腾讯财经",
                "rows": int(len(history_data)),
                "date_range": [history_data["日期"].iloc[0], history_data["日期"].iloc[-1]],
                "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            file_path, metadata_file = save_benchmark_atomically(
                history_data, Path(market_directory) / definition["file_name"], metadata
            )
            results.append(
                {
                    "名称": name,
                    "腾讯代码": definition["market_code"],
                    "行情来源": "腾讯财经 day",
                    "status": "success",
                    "file": str(file_path),
                    "metadata_file": str(metadata_file),
                    "rows": len(history_data),
                }
            )
        except Exception as error:
            results.append(
                {
                    "名称": name,
                    "腾讯代码": definition["market_code"],
                    "status": "failed",
                    "message": f"腾讯指数日线下载失败：{error}。未伪造或写入数据。",
                }
            )
    return results


def main():
    """只有用户显式执行本文件时才会请求腾讯指数接口。"""
    results = download_benchmarks()
    for result in results:
        print(result)
    return 0 if all(result["status"] == "success" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
