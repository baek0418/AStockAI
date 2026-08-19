"""将 v5.1 滚动样本外概率接入组合回测，严格延后一日执行。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio_backtest import PortfolioConfig, run_portfolio_backtest, save_backtest_report
from prediction_features_v2 import (
    DEFAULT_BENCHMARK,
    FEATURE_COLUMNS_V2,
    LABEL_COLUMN_V2,
    build_feature_dataset_v2,
    get_labeled_dataset_v2,
    load_benchmark_feature_frame,
)
from prediction_model import evaluate_rolling_windows, get_sklearn_dependencies


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
PROBABILITY_COLUMN = "预测跑赢基准概率"


def create_execution_signal_panel(feature_dataset, out_of_sample_frames, rebalance_interval=1):
    """把 t 日样本外概率移至下一交易日，形成完整持仓估值面板。

    特征在 t 日收盘后可得，因此绝不以 t 日收盘价成交。交易日在下一全市场
    交易日；若某股票没有该日价格则拒绝回测，不静默删掉无法成交的信号。
    """
    if rebalance_interval < 1:
        raise ValueError("rebalance_interval 必须至少为 1。")
    prices = (
        feature_dataset[["日期", "股票名称", "收盘"]]
        .dropna(subset=["日期", "股票名称", "收盘"])
        .drop_duplicates(["日期", "股票名称"])
        .sort_values(["日期", "股票名称"])
        .reset_index(drop=True)
    )
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices = prices.dropna(subset=["日期"])
    if not out_of_sample_frames:
        raise ValueError("没有可导出的滚动样本外信号。")
    signals = pd.concat(out_of_sample_frames, ignore_index=True)
    required = {"日期", "股票名称", PROBABILITY_COLUMN}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"样本外信号缺少字段：{sorted(missing)}。")
    signals = signals[["日期", "股票名称", PROBABILITY_COLUMN]].copy()
    signals["日期"] = pd.to_datetime(signals["日期"], errors="coerce")
    signals[PROBABILITY_COLUMN] = pd.to_numeric(signals[PROBABILITY_COLUMN], errors="coerce")
    signals = signals.dropna().drop_duplicates(["日期", "股票名称"])
    if signals.empty:
        raise ValueError("样本外概率为空，无法生成组合信号。")

    calendar = pd.Index(sorted(prices["日期"].unique()))
    next_dates = dict(zip(calendar[:-1], calendar[1:]))
    source_signal_count = len(signals)
    signals["信号日期"] = signals["日期"]
    signals["日期"] = signals["日期"].map(next_dates)
    signals = signals.dropna(subset=["日期"])
    if signals.empty:
        raise ValueError("样本外信号没有可用的下一交易日，无法回测。")
    signals = signals.rename(columns={"股票名称": "股票", PROBABILITY_COLUMN: "信号"})
    execution_dates = set(signals["日期"])
    rebalance_dates = set(sorted(execution_dates)[::rebalance_interval])
    start_date = min(execution_dates)
    end_date = max(execution_dates)
    period_dates = pd.Index([date for date in calendar if start_date <= date <= end_date])
    price_period = prices[prices["日期"].isin(period_dates)]
    expected_rows = len(period_dates)
    coverage = price_period.groupby("股票名称")["日期"].nunique()
    active_symbols = sorted(coverage[coverage == expected_rows].index)
    excluded_symbols = sorted(set(signals["股票"]) - set(active_symbols))
    signals = signals[signals["股票"].isin(active_symbols)].copy()
    if signals.empty:
        raise ValueError("没有股票覆盖完整的样本外执行区间，拒绝生成回测。")
    panel = price_period[price_period["股票名称"].isin(active_symbols)].rename(columns={"股票名称": "股票"})
    panel = panel.merge(
        signals[["日期", "股票", "信号", "信号日期"]],
        on=["日期", "股票"],
        how="left",
        validate="one_to_one",
    )
    expected = len(signals)
    actual = int(panel["信号"].notna().sum())
    if actual != expected:
        raise ValueError("完整覆盖股票中仍有样本外信号缺少执行价格，拒绝生成回测。")
    panel["调仓"] = panel["日期"].isin(rebalance_dates)
    panel = panel.sort_values(["日期", "股票"]).reset_index(drop=True)
    panel.attrs["执行信号数"] = int(panel.loc[panel["调仓"], "信号"].notna().sum())
    panel.attrs["调仓日数"] = len(rebalance_dates)
    panel.attrs["末日无执行价格而丢弃的信号数"] = source_signal_count - len(signals)
    panel.attrs["因价格覆盖不足排除的股票"] = excluded_symbols
    return panel


def run_oos_portfolio_research(project_directory=PROJECT_DIRECTORY, benchmark_name=DEFAULT_BENCHMARK, config=None):
    """生成并回测 v5.1 严格样本外组合信号，不训练最终全样本模型。"""
    project_directory = Path(project_directory)
    config = config or PortfolioConfig()
    try:
        dependencies = get_sklearn_dependencies()
        features, skipped_files = build_feature_dataset_v2(
            project_directory=project_directory, benchmark_name=benchmark_name
        )
        dataset = get_labeled_dataset_v2(features)
    except (RuntimeError, ValueError) as error:
        return {"status": "failed", "message": str(error)}
    evaluation = evaluate_rolling_windows(
        dataset,
        dependencies,
        feature_columns=FEATURE_COLUMNS_V2,
        label_column=LABEL_COLUMN_V2,
        probability_column=PROBABILITY_COLUMN,
        baseline_probability_column="基准跑赢概率",
        baseline_label="永远预测正类比例基线",
        baseline_probability_display_key="预测正类概率",
        outcome_description="跑赢市场基准",
        actual_positive_rate_key="实际跑赢基准率",
        out_of_sample_columns=["收盘"],
    )
    if not evaluation.get("ready"):
        return {"status": "insufficient", "message": evaluation["message"], "evaluation": evaluation}
    try:
        signal_panel = create_execution_signal_panel(
            features,
            evaluation["out_of_sample"],
            rebalance_interval=config.rebalance_interval,
        )
        benchmark = load_benchmark_feature_frame(
            project_directory / "data" / "market", benchmark_name
        )[["日期", "收盘"]]
        nav_data, trades_data, statistics = run_portfolio_backtest(signal_panel, config, benchmark)
    except ValueError as error:
        return {"status": "failed", "message": str(error), "evaluation": evaluation}

    output_directory = project_directory / "output" / "portfolio"
    nav_file, trades_file, report_file = save_backtest_report(
        nav_data, trades_data, statistics, output_directory, config
    )
    signal_file = output_directory / f"oos_portfolio_signals_{datetime.now():%Y-%m-%d}.csv"
    signal_panel.to_csv(signal_file, index=False, encoding="utf-8-sig")
    metadata_file = output_directory / f"oos_portfolio_research_{datetime.now():%Y-%m-%d}.json"
    metadata_file.write_text(
        json.dumps(
            {
                "策略": "v5.1 滚动样本外概率 Top-N 组合",
                "市场基准": benchmark_name,
                "信号执行规则": "t 日收盘后生成信号，下一交易日按收盘价与成本模型执行。",
                "滚动窗口": evaluation["windows"],
                "跳过文件": skipped_files,
                "信号覆盖诊断": {
                    "执行信号数": signal_panel.attrs.get("执行信号数"),
                    "调仓日数": signal_panel.attrs.get("调仓日数"),
                    "因价格覆盖不足排除的股票": signal_panel.attrs.get("因价格覆盖不足排除的股票", []),
                },
                "参数": config.__dict__,
                "组合统计": statistics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "status": "success",
        "message": "v5.1 滚动样本外组合回测完成，仅供研究。",
        "statistics": statistics,
        "signal_file": str(signal_file),
        "nav_file": str(nav_file),
        "trades_file": str(trades_file),
        "report_file": str(report_file),
        "metadata_file": str(metadata_file),
    }


def main():
    parser = argparse.ArgumentParser(description="AStockAI v5.1 滚动样本外组合研究")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--min-signal", type=float)
    arguments = parser.parse_args()
    config = PortfolioConfig(
        max_positions=arguments.max_positions,
        rebalance_interval=arguments.rebalance_interval,
        min_signal=arguments.min_signal,
    )
    result = run_oos_portfolio_research(benchmark_name=arguments.benchmark, config=config)
    print(result["message"])
    if result.get("statistics"):
        print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"success", "insufficient"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
