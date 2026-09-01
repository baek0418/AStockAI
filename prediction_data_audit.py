"""v5.1 预测研究的数据范围只读审计。"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from prediction_features import (
    HORIZON_DAYS,
    REQUIRED_COLUMNS,
    build_feature_dataset,
    create_stock_code_lookup,
    get_enabled_research_stock_codes,
    get_labeled_dataset,
)


def audit_data(data_directory, project_directory):
    """只扫描 data 根目录正式 CSV，不递归、不下载、不写入 CSV。"""
    data_directory = Path(data_directory)
    project_directory = Path(project_directory)
    code_lookup = create_stock_code_lookup(project_directory)
    allowed_stock_codes = get_enabled_research_stock_codes(project_directory)
    files = []
    for csv_file in sorted(data_directory.glob("*.csv")):
        stock_name = csv_file.stem.replace("历史", "")
        stock_code = code_lookup.get(stock_name, "")
        try:
            raw = pd.read_csv(csv_file, encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            files.append({"文件": csv_file.name, "状态": "无效", "原因": str(error)})
            continue
        missing_columns = REQUIRED_COLUMNS.difference(raw.columns)
        if missing_columns:
            files.append({"文件": csv_file.name, "状态": "无效", "原因": f"缺少字段：{sorted(missing_columns)}"})
            continue
        dates = pd.to_datetime(raw["日期"], errors="coerce")
        close = pd.to_numeric(raw["收盘"], errors="coerce")
        volume = pd.to_numeric(raw["成交量"], errors="coerce")
        valid = dates.notna() & close.notna() & volume.notna()
        valid_count = int(valid.sum())
        status = "可训练" if valid_count > HORIZON_DAYS else "数据不足"
        if allowed_stock_codes is not None and stock_code not in allowed_stock_codes:
            status = "研究池外"
        files.append(
            {
                "文件": csv_file.name,
                "股票名称": stock_name,
                "状态": status if valid_count else "无效",
                "原始行数": int(len(raw)),
                "有效样本数": valid_count,
                "日期范围": [
                    dates[valid].min().strftime("%Y-%m-%d") if valid_count else None,
                    dates[valid].max().strftime("%Y-%m-%d") if valid_count else None,
                ],
                "缺失比例": round(float(1 - valid_count / len(raw)), 6) if len(raw) else 1.0,
            }
        )
    try:
        feature_data, skipped = build_feature_dataset(
            data_directory, project_directory, allowed_stock_codes=allowed_stock_codes
        )
        labeled = get_labeled_dataset(feature_data)
        sample_distribution = [
            {"股票名称": name, "训练样本数": int(len(group))}
            for name, group in labeled.groupby("股票名称")
        ]
    except ValueError as error:
        skipped = [{"reason": str(error)}]
        sample_distribution = []
    valid_stocks = [file for file in files if file.get("状态") == "可训练"]
    valid_stock_count = len(valid_stocks)
    return {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据范围": (
            "仅使用当前启用的研究股票池快照中 data/*.csv；不包含 data/on_demand/ 或 data/market/。"
            if allowed_stock_codes is not None
            else "仅 data/*.csv；不包含 data/on_demand/ 或 data/market/。"
        ),
        "股票数量": valid_stock_count,
        "横截面限制": (
            f"当前仅有 {valid_stock_count} 只可训练股票，股票横截面覆盖有限，不能代表全市场。"
        ),
        "文件审计": files,
        "训练样本按股票分布": sample_distribution,
        "特征构建跳过文件": skipped,
    }


def create_markdown(audit):
    """生成审计 Markdown，突出覆盖有限和只读边界。"""
    lines = [
        "# AStockAI 预测研究数据审计",
        "",
        f"生成时间：{audit['生成时间']}",
        f"数据范围：{audit['数据范围']}",
        f"有效股票数量：{audit['股票数量']}",
        f"限制：{audit['横截面限制']}",
        "",
        "## 每只股票数据质量",
        "",
        "| 文件 | 状态 | 有效样本数 | 日期范围 | 缺失比例 |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for file in audit["文件审计"]:
        lines.append(
            f"| {file['文件']} | {file['状态']} | {file.get('有效样本数', 0)} | "
            f"{' 至 '.join(item or '无' for item in file.get('日期范围', []))} | {file.get('缺失比例', '无')} |"
        )
    lines.extend(["", "## 训练样本按股票分布", ""])
    for item in audit["训练样本按股票分布"]:
        lines.append(f"- {item['股票名称']}：{item['训练样本数']}")
    return "\n".join(lines) + "\n"


def run_audit(project_directory=None):
    """保存预测数据审计的 JSON 和 Markdown。"""
    project = Path(project_directory or Path(__file__).parent)
    audit = audit_data(project / "data", project)
    output_directory = project / "output" / "prediction"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().strftime("%Y-%m-%d")
    json_file = output_directory / f"prediction_data_audit_{report_date}.json"
    markdown_file = output_directory / f"prediction_data_audit_{report_date}.md"
    json_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_file.write_text(create_markdown(audit), encoding="utf-8")
    return audit, json_file, markdown_file


if __name__ == "__main__":
    _, json_file, markdown_file = run_audit()
    print(f"数据审计 JSON：{json_file}")
    print(f"数据审计 Markdown：{markdown_file}")
