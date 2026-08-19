"""更新统一股票池中的真实历史行情数据。"""

import time
from pathlib import Path

import pandas as pd
import requests

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


def save_stock(name, code, data_directory):
    """保存单只股票的真实行情数据，空数据时不创建 CSV 文件。"""
    print("===================")
    print("正在获取:", name)

    raw_data = get_stock(code)
    print("获取数据:", len(raw_data))

    history_data = create_history_dataframe(raw_data)
    data_file = data_directory / f"{name}历史.csv"
    history_data.to_csv(data_file, index=False, encoding="utf-8-sig")

    print(name, "保存成功", len(history_data), "条")
    return data_file


def run_update_data():
    """更新合并股票池中的股票，单只失败不阻断其他股票。"""
    project_directory = Path(__file__).parent
    data_directory = project_directory / "data"
    data_directory.mkdir(exist_ok=True)

    try:
        stock_universe = get_enabled_stock_universe()
    except (FileNotFoundError, ValueError) as error:
        print("股票池读取失败:")
        print(error)
        return {
            "success": False,
            "status": "failed",
            "message": "股票池读取失败",
            "output_file": None,
            "details": {"success_count": 0, "failed_count": 0, "failed_stocks": []},
        }

    success_stocks = []
    failed_stocks = []

    for stock in stock_universe:
        try:
            data_file = save_stock(
                stock["name"],
                stock["market_code"],
                data_directory,
            )
            success_stocks.append(
                {
                    "code": stock["code"],
                    "name": stock["name"],
                    "output_file": str(data_file),
                }
            )
            time.sleep(1)
        except Exception as error:
            print(stock["name"], "失败:")
            print(error)
            failed_stocks.append(
                {
                    "code": stock["code"],
                    "name": stock["name"],
                    "error": str(error),
                }
            )

    print("===================")
    print("全部更新完成")
    print("===================")

    if not failed_stocks:
        status = "success"
        message = "行情更新完成"
    elif success_stocks:
        status = "partial"
        message = "行情部分更新完成"
    else:
        status = "failed"
        message = "行情更新失败"

    return {
        "success": bool(success_stocks),
        "status": status,
        "message": message,
        "output_file": None,
        "details": {
            "success_count": len(success_stocks),
            "failed_count": len(failed_stocks),
            "success_stocks": success_stocks,
            "failed_stocks": failed_stocks,
        },
    }


if __name__ == "__main__":
    run_update_data()
