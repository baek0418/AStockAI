"""CSI 300 横截面动量的独立研究实验，不接入预测模型或产品页面。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from oos_portfolio_research import PROBABILITY_COLUMN, create_execution_signal_panel
from portfolio_backtest import PortfolioConfig, run_portfolio_backtest, save_backtest_report
from prediction_features import get_enabled_research_stock_codes
from prediction_features_v2 import DEFAULT_BENCHMARK, build_feature_dataset_v2, load_benchmark_feature_frame
from research_universe import CONFIG_FILE


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
LOOKBACK_DAYS = 252
SKIP_RECENT_DAYS = 21
MIN_CROSS_SECTION = 200
DEFAULT_REBALANCE_INTERVAL = 20
DEFAULT_POSITIONS = 10
SOURCE = "Jegadeesh and Titman (1993), Returns to Buying Winners and Selling Losers"
SOURCE_URL = "https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf"


def build_cross_sectional_momentum_signals(
    feature_dataset,
    lookback_days=LOOKBACK_DAYS,
    skip_recent_days=SKIP_RECENT_DAYS,
    min_cross_section=MIN_CROSS_SECTION,
):
    """用 t 日及以前价格构建 12-1 月横截面排名，不读取未来价格。"""
    if lookback_days <= skip_recent_days or skip_recent_days < 1:
        raise ValueError("动量回看期必须大于跳过期，且跳过期至少为 1。")
    prices = (
        pd.DataFrame(feature_dataset)[["日期", "股票名称", "收盘"]]
        .dropna(subset=["日期", "股票名称", "收盘"])
        .drop_duplicates(["日期", "股票名称"], keep="last")
        .copy()
    )
    prices["日期"] = pd.to_datetime(prices["日期"], errors="coerce")
    prices["收盘"] = pd.to_numeric(prices["收盘"], errors="coerce")
    prices = prices.dropna(subset=["日期", "收盘"])
    if prices.empty or (prices["收盘"] <= 0).any():
        raise ValueError("动量研究需要非空且为正数的收盘价。")

    # ``shift`` 必须沿全市场交易日历而不是每只股票自身的行数进行。否则某只
    # 股票漏了一天时，所谓 252 日回看会悄悄变为更早的日期，产生不可复核的信号。
    calendar = pd.Index(sorted(prices["日期"].unique()), name="日期")
    close_panel = prices.pivot(index="日期", columns="股票名称", values="收盘").reindex(calendar)
    momentum_panel = close_panel.shift(skip_recent_days) / close_panel.shift(lookback_days) - 1
    signals = (
        momentum_panel.rename_axis(index="日期", columns="股票名称")
        .reset_index()
        .melt(id_vars="日期", var_name="股票名称", value_name="12-1月动量")
        .dropna(subset=["12-1月动量"])
        .merge(prices, on=["日期", "股票名称"], how="left", validate="one_to_one")
    )
    coverage = signals.groupby("日期")["股票名称"].transform("size")
    signals = signals[coverage >= min_cross_section].copy()
    if signals.empty:
        raise ValueError("没有日期满足动量信号所需的横截面覆盖。")
    signals["信号"] = signals.groupby("日期")["12-1月动量"].rank(method="first", pct=True)
    return signals[["日期", "股票名称", "收盘", "12-1月动量", "信号"]].sort_values(
        ["日期", "股票名称"]
    ).reset_index(drop=True)


def _portfolio_config(max_positions, rebalance_interval):
    return PortfolioConfig(
        max_positions=max_positions,
        max_weight=1 / max_positions,
        rebalance_interval=rebalance_interval,
        target_annual_volatility=0.18,
        market_ma_window=20,
        risk_off_exposure=0.0,
    )


def _sensitivity_result(signal_panel, benchmark, rebalance_interval, positions):
    config = _portfolio_config(positions, rebalance_interval)
    _, _, statistics = run_portfolio_backtest(signal_panel, config, benchmark)
    return {"最大持仓数": positions, "统计": statistics}


def calculate_temporal_windows(signal_panel, benchmark, config, window_count=3):
    """按预先存在的调仓日切分连续时间段；因子不拟合参数，分段只用于稳定性审阅。"""
    rebalance_dates = pd.Index(
        sorted(pd.to_datetime(signal_panel.loc[signal_panel["调仓"], "日期"].unique()))
    )
    if len(rebalance_dates) < window_count:
        return []
    boundaries = [len(rebalance_dates) * index // window_count for index in range(window_count + 1)]
    results = []
    for index, (start_index, end_index) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        start = rebalance_dates[start_index]
        next_start = rebalance_dates[end_index] if end_index < len(rebalance_dates) else None
        window_panel = signal_panel[signal_panel["日期"] >= start].copy()
        if next_start is not None:
            window_panel = window_panel[window_panel["日期"] < next_start]
        window_dates = pd.to_datetime(window_panel["日期"].drop_duplicates().sort_values())
        if window_panel.empty or window_dates.empty:
            continue
        window_benchmark = pd.DataFrame(benchmark).copy()
        window_benchmark["日期"] = pd.to_datetime(window_benchmark["日期"], errors="coerce")
        window_benchmark = window_benchmark[window_benchmark["日期"].isin(window_dates)]
        _, _, statistics = run_portfolio_backtest(window_panel, config, window_benchmark)
        results.append(
            {
                "窗口": index,
                "日期范围": [str(window_dates.iloc[0].date()), str(window_dates.iloc[-1].date())],
                "统计": statistics,
            }
        )
    return results


def run_momentum_factor_research(
    project_directory=PROJECT_DIRECTORY,
    benchmark_name=DEFAULT_BENCHMARK,
    rebalance_interval=DEFAULT_REBALANCE_INTERVAL,
    max_positions=DEFAULT_POSITIONS,
):
    """运行预先固定参数的长仓动量研究并保存可复核产物。"""
    project_directory = Path(project_directory)
    if get_enabled_research_stock_codes(project_directory) is None:
        return {"status": "failed", "message": "未启用版本化研究股票池，拒绝运行动量实验。"}
    try:
        features, skipped_files = build_feature_dataset_v2(
            project_directory=project_directory, benchmark_name=benchmark_name
        )
        signals = build_cross_sectional_momentum_signals(features)
        execution_signals = signals.rename(columns={"信号": PROBABILITY_COLUMN})[
            ["日期", "股票名称", PROBABILITY_COLUMN]
        ]
        signal_panel = create_execution_signal_panel(
            features,
            [execution_signals],
            rebalance_interval=rebalance_interval,
        )
        benchmark = load_benchmark_feature_frame(
            project_directory / "data" / "market", benchmark_name
        )[["日期", "收盘"]]
        config = _portfolio_config(max_positions, rebalance_interval)
        nav_data, trades_data, statistics = run_portfolio_backtest(signal_panel, config, benchmark)
    except (RuntimeError, ValueError) as error:
        return {"status": "failed", "message": str(error)}

    output_directory = project_directory / "output" / "momentum"
    nav_file, trades_file, backtest_file = save_backtest_report(
        nav_data, trades_data, statistics, output_directory, config
    )
    report_date = datetime.now().strftime("%Y-%m-%d")
    signal_file = output_directory / f"momentum_signals_{report_date}.csv"
    signals.to_csv(signal_file, index=False, encoding="utf-8-sig")
    sensitivity = [
        _sensitivity_result(signal_panel, benchmark, rebalance_interval, positions)
        for positions in (5, 10, 20)
    ]
    temporal_windows = calculate_temporal_windows(signal_panel, benchmark, config)
    stable_windows = temporal_windows and all(
        window["统计"].get("超额累计收益率", float("-inf")) > 0 for window in temporal_windows
    )
    report = {
        "实验": "CSI 300 12-1 月横截面动量（长仓）",
        "假设": "过去 12 个月、跳过最近 1 个月的相对强势可提供后续相对收益排序信息。",
        "研究来源": {"文献": SOURCE, "链接": SOURCE_URL},
        "状态": "仅研究；不得接入日报、Web 或自动交易。",
        "数据范围": {
            "研究股票池配置": str(CONFIG_FILE.relative_to(PROJECT_DIRECTORY)),
            "股票数量": int(features["股票代码"].nunique()),
            "信号日期范围": [str(signals["日期"].min().date()), str(signals["日期"].max().date())],
            "跳过文件": skipped_files,
            "幸存者偏差": "使用当前 CSI 300 成分股快照回溯历史；未使用历史成分变更，结果存在幸存者偏差。",
        },
        "特征定义": {
            "公式": "close[t-21] / close[t-252] - 1，再于每个交易日横截面排序。",
            "回看交易日": LOOKBACK_DAYS,
            "跳过最近交易日": SKIP_RECENT_DAYS,
            "最小横截面股票数": MIN_CROSS_SECTION,
            "可用时点": "t 日收盘后；信号映射至下一交易日执行。",
        },
        "组合规则": {
            "长仓TopN": max_positions,
            "调仓间隔交易日": rebalance_interval,
            "交易约束": "次日执行、整手、佣金、印花税、滑点；停牌及涨跌停仅可由显式标记模拟。",
            "市场风控": "20 日均线风险关闭与 18% 年化波动率目标，只用调仓日前基准价格。",
        },
        "主实验统计": statistics,
        "敏感性分析": sensitivity,
        "连续时间分段结果": temporal_windows,
        "基线对照": "v5.1 OHLCV 概率模型当前未通过样本外准入；本实验为独立价格因子基线，不能据此宣布改进。",
        "准入结论": (
            "三个连续时间段均有正超额收益，但仍须以历史成分股和更长数据独立复验；当前一律不准入产品。"
            if stable_windows
            else "连续时间段表现不稳定，且须以历史成分股和更长数据独立复验；当前一律不准入产品。"
        ),
    }
    report_file = output_directory / f"momentum_factor_research_{report_date}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "success",
        "message": "动量因子研究完成，仅供研究。",
        "statistics": statistics,
        "signal_file": str(signal_file),
        "nav_file": str(nav_file),
        "trades_file": str(trades_file),
        "backtest_file": str(backtest_file),
        "report_file": str(report_file),
    }


def main():
    parser = argparse.ArgumentParser(description="AStockAI CSI 300 12-1 月横截面动量研究")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--rebalance-interval", type=int, default=DEFAULT_REBALANCE_INTERVAL)
    parser.add_argument("--max-positions", type=int, default=DEFAULT_POSITIONS)
    arguments = parser.parse_args()
    result = run_momentum_factor_research(
        benchmark_name=arguments.benchmark,
        rebalance_interval=arguments.rebalance_interval,
        max_positions=arguments.max_positions,
    )
    print(result["message"])
    if result.get("statistics"):
        print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
