"""导入已从可信来源导出的沪深300 CSV，并安全启用研究股票池。"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from research_universe import CONFIG_FILE, enable_snapshot, save_csi300_snapshot


CODE_HEADERS = {"code", "股票代码", "证券代码", "成分券代码", "品种代码"}
NAME_HEADERS = {"name", "股票名称", "证券名称", "成分券名称", "品种名称"}


def read_csv_rows(csv_file):
    """兼容中证官网常见的 UTF-8 与 GBK/GB18030 CSV 导出。"""
    path = Path(csv_file)
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("CSV 编码无法识别，请另存为 UTF-8 或 GBK CSV。")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or not rows[0]:
        raise ValueError("CSV 没有可读取的数据行。")
    headers = {header.strip(): header for header in rows[0] if header}
    code_header = next((headers[header] for header in CODE_HEADERS if header in headers), None)
    name_header = next((headers[header] for header in NAME_HEADERS if header in headers), None)
    if not code_header or not name_header:
        raise ValueError("CSV 需要包含股票代码和股票名称两列。")
    return [
        {"code": row.get(code_header, ""), "name": row.get(name_header, "")}
        for row in rows
        if row
    ]


def import_snapshot(csv_file, as_of_date=None, config_file=CONFIG_FILE):
    """导入、完整性校验、落盘并启用，任一步失败均不改变现有配置。"""
    config_path = Path(config_file)
    stocks = read_csv_rows(csv_file)
    snapshot = save_csi300_snapshot(
        stocks,
        config_path.parent / "universe",
        now=datetime.now(),
        as_of_date=as_of_date,
        source="用户导入的沪深300成分股 CSV",
        source_note="请在导入前确认导出日期与来源；建议使用中证指数官网或已授权数据服务。",
    )
    enable_snapshot(snapshot, config_path)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description="导入沪深300成分股 CSV 并启用研究股票池")
    parser.add_argument("csv_file", help="包含股票代码、股票名称两列的 CSV 文件")
    parser.add_argument("--as-of-date", help="成分股生效/导出日期，格式 YYYY-MM-DD")
    arguments = parser.parse_args()
    try:
        snapshot = import_snapshot(arguments.csv_file, arguments.as_of_date)
    except (OSError, ValueError) as error:
        print(f"导入失败：{error}")
        return 1
    print(f"沪深300研究股票池已启用：{snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
