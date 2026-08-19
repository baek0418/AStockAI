"""将本地正式 A 股日线转换为可审计的标准长表。

该模块只读取 ``data/*.csv``，不会下载行情、修改源 CSV 或替换现有预测流程。
输出表的字段名与 Qlib 常用日线字段保持一致，但尚未声称数据已经满足 Qlib
训练要求；代码缺失、复权方式未知和价格异常都会留在审计报告中。
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLUMNS = {"日期", "开盘", "收盘", "最高", "最低", "成交量"}
STANDARD_COLUMNS = [
    "instrument",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "adjustment",
    "source_file",
    "source_file_modified_at",
]
LOCAL_SOURCE = "AStockAI 本地正式历史 CSV"
UNKNOWN_ADJUSTMENT = "unknown"


def load_stock_code_lookup(project_directory):
    """从既有股票池读取名称到代码的映射；读取失败时保持适配器可用。"""
    try:
        from stock_universe import get_enabled_stock_universe

        return {stock["name"]: stock["code"] for stock in get_enabled_stock_universe()}
    except (FileNotFoundError, ImportError, ValueError):
        return {}


def make_instrument(stock_name, stock_code):
    """生成 Qlib 风格的稳定标识；代码未知时使用显式不可训练标识。"""
    code = str(stock_code or "").strip()
    if len(code) == 6 and code.isdigit():
        return f"sh{code}" if code.startswith("6") else f"sz{code}"
    return f"unknown_{stock_name}"


def read_history_csv(history_file):
    """读取并校验单个正式历史 CSV，返回规范化原始日线与质量信息。"""
    history_file = Path(history_file)
    try:
        raw = pd.read_csv(history_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(history_file)
    missing = REQUIRED_SOURCE_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"缺少字段：{'、'.join(sorted(missing))}。")

    history = raw.copy()
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    for column in ("开盘", "收盘", "最高", "最低", "成交量"):
        history[column] = pd.to_numeric(history[column], errors="coerce")

    invalid_required = history[["日期", "开盘", "收盘", "最高", "最低", "成交量"]].isna().any(axis=1)
    invalid_values = (
        (history["开盘"] <= 0)
        | (history["收盘"] <= 0)
        | (history["最高"] <= 0)
        | (history["最低"] <= 0)
        | (history["成交量"] < 0)
    )
    invalid_ohlc = (history["最高"] < history[["开盘", "收盘", "最低"]].max(axis=1)) | (
        history["最低"] > history[["开盘", "收盘", "最高"]].min(axis=1)
    )
    invalid = invalid_required | invalid_values | invalid_ohlc
    valid = history.loc[~invalid].copy()
    duplicate_dates = int(valid.duplicated("日期", keep="last").sum())
    valid = valid.sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True)
    return valid, {
        "原始行数": int(len(raw)),
        "无效行数": int(invalid.sum()),
        "重复日期行数": duplicate_dates,
    }


def find_date_gaps(dates, threshold_days=7):
    """仅报告较长自然日间隔；它是提示，不将节假日自动认定为缺失。"""
    if len(dates) < 2:
        return []
    date_series = pd.Series(pd.to_datetime(dates)).sort_values().reset_index(drop=True)
    gaps = date_series.diff().dt.days
    return [
        {
            "前一交易日": date_series.iloc[index - 1].strftime("%Y-%m-%d"),
            "后一交易日": date_series.iloc[index].strftime("%Y-%m-%d"),
            "自然日间隔": int(gap),
        }
        for index, gap in gaps.items()
        if pd.notna(gap) and gap > threshold_days
    ]


def adapt_history_file(history_file, stock_name, stock_code=""):
    """适配一只股票，并保留来源、文件修改时间与复权未知状态。"""
    history_file = Path(history_file)
    history, quality = read_history_csv(history_file)
    if history.empty:
        raise ValueError("没有通过 OHLCV 校验的有效日线。")
    modified_at = datetime.fromtimestamp(history_file.stat().st_mtime, tz=timezone.utc).isoformat()
    adapted = pd.DataFrame(
        {
            "instrument": make_instrument(stock_name, stock_code),
            "date": history["日期"].dt.strftime("%Y-%m-%d"),
            "open": history["开盘"].astype(float),
            "high": history["最高"].astype(float),
            "low": history["最低"].astype(float),
            "close": history["收盘"].astype(float),
            "volume": history["成交量"].astype(float),
            "source": LOCAL_SOURCE,
            "adjustment": UNKNOWN_ADJUSTMENT,
            "source_file": history_file.name,
            "source_file_modified_at": modified_at,
        }
    )
    quality.update(
        {
            "文件": history_file.name,
            "股票名称": stock_name,
            "股票代码": str(stock_code or ""),
            "instrument": adapted["instrument"].iloc[0],
            "有效行数": int(len(adapted)),
            "日期范围": [adapted["date"].iloc[0], adapted["date"].iloc[-1]],
            "长日期间隔": find_date_gaps(history["日期"]),
            "复权方式": UNKNOWN_ADJUSTMENT,
            "可用于 Qlib": bool(stock_code),
        }
    )
    return adapted[STANDARD_COLUMNS], quality


def build_standard_market_data(data_directory, project_directory=None, code_lookup=None):
    """只适配 data 根目录 CSV，递归目录、市场基准和按需缓存均不在范围内。"""
    data_directory = Path(data_directory)
    if code_lookup is None:
        code_lookup = load_stock_code_lookup(project_directory)
    frames = []
    files = []
    for history_file in sorted(data_directory.glob("*.csv")):
        stock_name = history_file.stem.replace("历史", "")
        try:
            frame, quality = adapt_history_file(history_file, stock_name, code_lookup.get(stock_name, ""))
            frames.append(frame)
            files.append({"状态": "已适配", **quality})
        except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
            files.append({"文件": history_file.name, "股票名称": stock_name, "状态": "跳过", "原因": str(error)})
    if not frames:
        raise ValueError("data 根目录中没有可适配的正式 OHLCV 历史 CSV。")
    dataset = pd.concat(frames, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)
    return dataset[STANDARD_COLUMNS], files


def create_audit(dataset, file_audit):
    """生成输出文件与数据范围均可追溯的审计信息。"""
    unknown_instruments = sorted(dataset.loc[dataset["instrument"].str.startswith("unknown_"), "instrument"].unique())
    return {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据范围": "仅 data/*.csv；不读取 data/market/、data/on_demand/ 或任何网络数据。",
        "标准字段": STANDARD_COLUMNS,
        "行数": int(len(dataset)),
        "标的数": int(dataset["instrument"].nunique()),
        "日期范围": [dataset["date"].min(), dataset["date"].max()],
        "数据源": LOCAL_SOURCE,
        "复权方式": UNKNOWN_ADJUSTMENT,
        "Qlib 准备状态": "未准备：复权方式尚未有可验证元数据。",
        "未知代码标的": unknown_instruments,
        "文件审计": file_audit,
    }


def create_markdown(audit):
    """输出方便人工审阅的简短 Markdown 审计报告。"""
    lines = [
        "# AStockAI 标准行情数据审计",
        "",
        f"生成时间：{audit['生成时间']}",
        f"数据范围：{audit['数据范围']}",
        f"行数：{audit['行数']}；标的数：{audit['标的数']}；日期范围：{' 至 '.join(audit['日期范围'])}",
        f"数据源：{audit['数据源']}；复权方式：{audit['复权方式']}",
        f"Qlib 准备状态：{audit['Qlib 准备状态']}",
        "",
        "## 文件质量",
        "",
        "| 文件 | 状态 | 有效行数 | 无效行数 | 重复日期行数 | 日期范围 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in audit["文件审计"]:
        lines.append(
            f"| {item['文件']} | {item['状态']} | {item.get('有效行数', 0)} | "
            f"{item.get('无效行数', 0)} | {item.get('重复日期行数', 0)} | "
            f"{' 至 '.join(item.get('日期范围', [])) or item.get('原因', '无')} |"
        )
    if audit["未知代码标的"]:
        lines.extend(["", "## 需要补全代码的标的", "", *[f"- {item}" for item in audit["未知代码标的"]]])
    return "\n".join(lines) + "\n"


def run_market_data_adapter(project_directory=None):
    """生成只读研究产物；不写入 data/ 或 models/。"""
    project = Path(project_directory or Path(__file__).parent)
    dataset, file_audit = build_standard_market_data(project / "data", project)
    audit = create_audit(dataset, file_audit)
    output_directory = project / "output" / "research"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().strftime("%Y-%m-%d")
    data_file = output_directory / f"standard_market_data_{report_date}.csv"
    json_file = output_directory / f"standard_market_data_audit_{report_date}.json"
    markdown_file = output_directory / f"standard_market_data_audit_{report_date}.md"
    dataset.to_csv(data_file, index=False, encoding="utf-8")
    json_file.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_file.write_text(create_markdown(audit), encoding="utf-8")
    return dataset, audit, data_file, json_file, markdown_file


if __name__ == "__main__":
    _, _, data_file, json_file, markdown_file = run_market_data_adapter()
    print(f"标准行情 CSV：{data_file}")
    print(f"数据审计 JSON：{json_file}")
    print(f"数据审计 Markdown：{markdown_file}")
