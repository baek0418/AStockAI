"""AStockAI v2.8 第一阶段：生成可追溯的量化事实快照。"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from score import calculate_score
from stock_universe import create_stock_code_lookup, get_enabled_stock_universe
from strategy_backtest import INITIAL_CAPITAL, run_strategy_backtest
from conservative_candidates import build_conservative_candidates
from technical_indicators import latest_technical_indicators


REQUIRED_COLUMNS = {"日期", "收盘", "成交量"}


def is_historical_data_file(file_path):
    """判断 CSV 文件是否包含量化分析所需的历史行情字段。"""
    try:
        columns = set(pd.read_csv(file_path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return False

    return REQUIRED_COLUMNS.issubset(columns)


def get_latest_market_data(file_path):
    """读取单只股票最新交易日的日期和收盘价，不重复计算技术指标。"""
    history_data = pd.read_csv(file_path)
    history_data["日期"] = pd.to_datetime(history_data["日期"], errors="coerce")
    history_data["收盘"] = pd.to_numeric(history_data["收盘"], errors="coerce")
    history_data = history_data.dropna(subset=["日期", "收盘"])
    latest_data = history_data.sort_values("日期").iloc[-1]

    return {
        "日期": latest_data["日期"].strftime("%Y-%m-%d"),
        "收盘价": float(latest_data["收盘"]),
    }


def create_stock_snapshot(file_path, stock_code_lookup):
    """调用既有评分程序一次，生成单只股票的量化事实记录。"""
    stock_name = file_path.stem.replace("历史", "")
    latest_market_data = get_latest_market_data(file_path)
    score_result = calculate_score(file_path)
    try:
        technical_indicators = latest_technical_indicators(pd.read_csv(file_path, encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError):
        technical_indicators = {"数据状态": "数据不足，未生成扩展技术指标。"}

    stock_snapshot = {
        "股票名称": stock_name,
        "数据文件": file_path.name,
        **latest_market_data,
        "RSI": score_result["RSI"],
        "MA5": score_result["MA5"],
        "MA20": score_result["MA20"],
        "MACD": score_result["MACD"],
        "综合评分": score_result["评分"],
        "建议": score_result["建议"],
        "扩展技术指标": technical_indicators,
    }

    stock_code = stock_code_lookup.get(stock_name)
    if stock_code:
        stock_snapshot["股票代码"] = stock_code

    return stock_snapshot


def collect_stock_snapshots(data_directory, stock_code_lookup):
    """扫描 data 文件夹，收集全部历史股票的量化事实记录。"""
    stock_snapshots = []

    for file_path in sorted(data_directory.glob("*.csv")):
        if not is_historical_data_file(file_path):
            continue

        stock_snapshots.append(create_stock_snapshot(file_path, stock_code_lookup))

    return stock_snapshots


def sort_stock_rankings(stock_snapshots):
    """按综合评分从高到低生成股票排行榜，同分时按名称排序。"""
    return sorted(
        stock_snapshots,
        key=lambda item: (-item["综合评分"], item["股票名称"]),
    )


def get_snapshot_date(stock_rankings):
    """取得股票池中最新的数据日期，作为量化事实快照日期。"""
    return max(item["日期"] for item in stock_rankings)


def create_backtest_snapshot():
    """运行既有策略回测，并整理报告可直接使用的真实回测结果。"""
    trades, equity_curve, statistics = run_strategy_backtest()

    return {
        "初始资金": INITIAL_CAPITAL,
        "策略说明": "每日选择股票池中评分最高且评分不低于70的股票，全仓持有5个后续交易日。",
        "统计": statistics,
        "交易记录": trades,
        "资金曲线文件": "equity_curve.csv",
        "资金曲线开始日期": equity_curve[0]["日期"],
        "资金曲线结束日期": equity_curve[-1]["日期"],
    }


def create_quant_snapshot(stock_rankings, backtest_snapshot, conservative_candidates=None):
    """组合股票排行榜和策略回测结果，创建完整的量化事实快照。"""
    snapshot_date = get_snapshot_date(stock_rankings)

    snapshot = {
        "快照日期": snapshot_date,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "股票数量": len(stock_rankings),
        "股票排行榜": stock_rankings,
        "策略回测": backtest_snapshot,
    }
    if conservative_candidates is not None:
        snapshot["稳健研究候选"] = conservative_candidates
    return snapshot


def save_quant_snapshot(quant_snapshot, output_directory):
    """创建 output 文件夹并将量化事实快照保存为 JSON 文件。"""
    output_directory.mkdir(exist_ok=True)
    snapshot_file = output_directory / (
        f"quant_snapshot_{quant_snapshot['快照日期']}.json"
    )

    with open(snapshot_file, "w", encoding="utf-8") as file:
        json.dump(quant_snapshot, file, ensure_ascii=False, indent=2)

    return snapshot_file


def run_research_data():
    """执行量化事实快照生成流程，并返回快照内容和保存路径。"""
    project_directory = Path(__file__).parent
    stock_universe = get_enabled_stock_universe()
    stock_code_lookup = create_stock_code_lookup(stock_universe)
    stock_snapshots = collect_stock_snapshots(
        project_directory / "data",
        stock_code_lookup,
    )

    if not stock_snapshots:
        raise ValueError("data 文件夹中没有可用的历史股票 CSV 数据。")

    stock_rankings = sort_stock_rankings(stock_snapshots)
    backtest_snapshot = create_backtest_snapshot()
    snapshot_date = get_snapshot_date(stock_rankings)
    conservative_candidates = build_conservative_candidates(
        stock_rankings,
        project_directory / "data",
        project_directory / "data" / "market",
        snapshot_date,
    )
    quant_snapshot = create_quant_snapshot(
        stock_rankings,
        backtest_snapshot,
        conservative_candidates,
    )
    snapshot_file = save_quant_snapshot(
        quant_snapshot,
        project_directory / "output",
    )

    print("量化事实快照生成成功:")
    print(snapshot_file)
    return quant_snapshot, snapshot_file


if __name__ == "__main__":
    run_research_data()
