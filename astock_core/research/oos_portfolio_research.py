"""将 v5.1 滚动样本外概率接入组合回测，严格延后一日执行。"""

import argparse
import json
import platform
from astock_core.data.prediction_benchmark_data import BENCHMARKS
from astock_core.research.historical_inputs import load_historical_inputs, build_historical_features, sha256
from datetime import datetime
from pathlib import Path

import pandas as pd

from astock_core.portfolio.portfolio_backtest import PortfolioConfig, run_portfolio_backtest, save_backtest_report
from astock_core.research.prediction_features_v2 import (
    DEFAULT_BENCHMARK,
    FEATURE_COLUMNS_V2,
    LABEL_COLUMN_V2,
    build_feature_dataset_v2,
    get_labeled_dataset_v2,
    load_benchmark_feature_frame,
)
from astock_core.research.prediction_model import evaluate_rolling_windows, get_sklearn_dependencies


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()
PROBABILITY_COLUMN = "预测跑赢基准概率"


def create_execution_signal_panel(feature_dataset, out_of_sample_frames, rebalance_interval=1,
                                  execution_prices=None, trading_calendar=None):
    """信号延后一交易日；从不按未来行情完整性删股，也不前填成交价。

    正式入口必须提供独立原始成交价格和交易日历。兼容的 feature_dataset
    价格仅用于单元测试/面板检查，不构成严格回测的价格依据。
    """
    if rebalance_interval < 1:
        raise ValueError("rebalance_interval 必须至少为 1。")
    prices = (execution_prices if execution_prices is not None else feature_dataset).copy()
    prices = prices.rename(columns={"股票名称": "股票"})
    prices["日期"] = pd.to_datetime(prices["日期"], errors="raise")
    if prices.duplicated(["日期", "股票"]).any():
        raise ValueError("成交价格存在重复日期/股票。")
    if not out_of_sample_frames:
        raise ValueError("没有可导出的滚动样本外信号。")
    signals = pd.concat(out_of_sample_frames, ignore_index=True)
    required = {"日期", "股票名称", PROBABILITY_COLUMN}
    if not required.issubset(signals):
        raise ValueError(f"样本外信号缺少字段：{sorted(required - set(signals))}。")
    signals = signals[["日期", "股票名称", PROBABILITY_COLUMN]].rename(columns={"股票名称": "股票", PROBABILITY_COLUMN: "信号"})
    signals["日期"] = pd.to_datetime(signals["日期"], errors="raise")
    if signals.duplicated(["日期", "股票"]).any() or signals["信号"].isna().any():
        raise ValueError("样本外信号重复或缺失，拒绝静默删行。")
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(
        trading_calendar if trading_calendar is not None else prices["日期"].unique()
    ).unique()))
    if not signals["日期"].isin(calendar).all():
        raise ValueError("信号日不在交易日历内。")
    next_dates = dict(zip(calendar[:-1], calendar[1:]))
    references = prices[["日期","股票","收盘"]].rename(columns={"收盘":"委托参考价"})
    signals = signals.merge(references,on=["日期","股票"],how="left",validate="one_to_one")
    if signals["委托参考价"].isna().any():
        raise ValueError("信号日缺少原始委托参考价格。")
    signals["信号日期"] = signals["日期"]
    signals["日期"] = signals["日期"].map(next_dates)
    terminal_count = int(signals["日期"].isna().sum())
    signals = signals.dropna(subset=["日期"])
    if signals.empty:
        raise ValueError("样本外信号没有可用的下一交易日，无法回测。")
    rebalance_dates = set(sorted(signals["日期"].unique())[::rebalance_interval])
    signals = signals[signals["日期"].isin(rebalance_dates)]
    # 保留整个已定义执行区间内的价格；价格缺失不改变候选股票名单。
    period = prices[prices["日期"].between(signals["日期"].min(), max(next_dates.values()))].copy()
    # 测试/兼容调用只回放到最后信号执行日；正式日历由研究入口限定评估终点。
    if trading_calendar is None:
        period = period[period["日期"] <= signals["日期"].max()]
    panel = period.merge(signals, on=["日期", "股票"], how="outer", validate="one_to_one")
    missing = panel.loc[panel["信号"].notna() & panel["收盘"].isna(), ["日期", "股票"]]
    if not missing.empty:
        raise ValueError(f"信号缺少执行价格，拒绝回测而非删股：{missing.to_dict('records')[:5]}。")
    panel["调仓"] = panel["日期"].isin(rebalance_dates)
    panel = panel.sort_values(["日期", "股票"]).reset_index(drop=True)
    panel.attrs["执行信号数"] = int(panel["信号"].notna().sum())
    panel.attrs["调仓日数"] = len(rebalance_dates)
    panel.attrs["末日无执行价格而丢弃的信号数"] = terminal_count
    panel.attrs["因价格覆盖不足排除的股票"] = []
    panel.attrs["完整性筛选"] = "禁用；执行或持仓估值缺失将明确拒绝回测。"
    return panel


def _save_research_record(project_directory, record):
    directory = Path(project_directory) / "output/portfolio"
    directory.mkdir(parents=True,exist_ok=True)
    target = directory / f"oos_portfolio_research_{datetime.now():%Y-%m-%d_%H%M%S_%f}.json"
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return str(target)


def _refuse_research(project_directory, message, config, input_audit=None):
    root = Path(project_directory)
    snapshot = root / 'config/research_universe.json'
    record = {
        "策略":"v5.1 OHLCV 架构基线：历史输入审计",
        "状态":"rejected", "说明":message, "组合统计":{}, "参数":config.__dict__,
        "审计":{"状态":"rejected", "是否策略准入":False,
                "待补证据":["逐交易日历史成分及当时可得时间", "原始成交/估值价格及显式买卖状态", "完整公司行为台账及派息/上市日期", "来源交易日历"],
                "当前股票池配置校验值":sha256(snapshot) if snapshot.exists() else None,
                "边界":"当前股票池快照不回填历史；正式前复权恢复不等于获得可执行历史价格。"},
    }
    if input_audit is not None:
        record["审计"]["已核验输入"] = input_audit
    file = _save_research_record(root,record)
    return {"status":"insufficient","message":message,"metadata_file":file,"audit":record['审计']}


def run_oos_portfolio_research(project_directory=PROJECT_DIRECTORY, benchmark_name=DEFAULT_BENCHMARK, config=None):
    """生成并回测 v5.1 严格样本外组合信号，不训练最终全样本模型。"""
    project_directory = Path(project_directory)
    config = config or PortfolioConfig()
    try:
        historical, input_audit = load_historical_inputs(project_directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return _refuse_research(project_directory,str(error),config,locals().get("input_audit"))
    try:
        dependencies = get_sklearn_dependencies()
        features, skipped_files = build_feature_dataset_v2(
            project_directory=project_directory, benchmark_name=benchmark_name,
            stock_features=build_historical_features(historical),
        )
        dataset = get_labeled_dataset_v2(features)
    except (RuntimeError, ValueError) as error:
        return _refuse_research(project_directory,str(error),config,locals().get("input_audit"))
    benchmark_path = project_directory / "data/market" / BENCHMARKS[benchmark_name]["file_name"]
    input_audit["输入校验值"]["benchmark"] = sha256(benchmark_path)
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
        return _refuse_research(project_directory,evaluation["message"],config,input_audit)
    try:
        signal_panel = create_execution_signal_panel(
            features,
            evaluation["out_of_sample"],
            rebalance_interval=config.rebalance_interval,
            execution_prices=historical["prices"].rename(columns={"股票代码":"股票"}),
            trading_calendar=historical["calendar"]["日期"],
        )
        benchmark = load_benchmark_feature_frame(
            project_directory / "data" / "market", benchmark_name
        )[["日期", "收盘"]]
        nav_data, trades_data, statistics = run_portfolio_backtest(signal_panel, config, benchmark, historical["corporate_actions"])
    except ValueError as error:
        return _refuse_research(project_directory,str(error),config,locals().get("input_audit"))

    output_directory = project_directory / "output" / "portfolio" / f"replay_{datetime.now():%Y%m%d_%H%M%S_%f}"
    nav_file, trades_file, report_file = save_backtest_report(
        nav_data, trades_data, statistics, output_directory, config
    )
    signal_file = output_directory / f"oos_portfolio_signals_{datetime.now():%Y-%m-%d}.csv"
    signal_panel.to_csv(signal_file, index=False, encoding="utf-8-sig")
    metadata_file = project_directory / "output/portfolio" / f"oos_portfolio_research_{datetime.now():%Y-%m-%d_%H%M%S_%f}.json"
    metadata_file.write_text(
        json.dumps(
            {
                "策略": "v5.1 OHLCV 架构基线：因果收益特征与原始价格回放",
                "状态":"research_only",
                "审计":input_audit,
                "基线说明":"保留原v5.1负结果；此处沿用模型结构，特征价格改为因果收益序列，不与旧口径收益直接排名。",
                "复现": {"Python":platform.python_version(), "pandas":pd.__version__, "scikit_learn":dependencies["sklearn"].__version__,
                         "代码校验值":{str(path.relative_to(PROJECT_DIRECTORY)):sha256(path) for path in [Path(__file__),PROJECT_DIRECTORY/"astock_core/portfolio/portfolio_backtest.py",PROJECT_DIRECTORY/"astock_core/research/historical_inputs.py", PROJECT_DIRECTORY/"astock_core/portfolio/corporate_actions.py", PROJECT_DIRECTORY/"astock_core/research/prediction_features.py", PROJECT_DIRECTORY/"astock_core/research/prediction_features_v2.py", PROJECT_DIRECTORY/"astock_core/research/prediction_model.py"]}},
                "市场基准": benchmark_name,
                "信号执行规则": "t 日收盘后生成信号，下一交易日按收盘价与成本模型执行。",
                "滚动窗口": evaluation["windows"],
                "跳过文件": skipped_files,
                "信号覆盖诊断": {
                    "执行信号数": signal_panel.attrs.get("执行信号数"),
                    "调仓日数": signal_panel.attrs.get("调仓日数"),
                    "因价格覆盖不足排除的股票": [],
                    "完整性筛选":"禁用；缺失执行价或持仓估值时拒绝回放",
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
    parser.add_argument("--target-volatility", type=float, default=0.18)
    parser.add_argument("--market-ma-window", type=int, default=20)
    parser.add_argument("--risk-off-exposure", type=float, default=0.0)
    arguments = parser.parse_args()
    config = PortfolioConfig(
        max_positions=arguments.max_positions,
        rebalance_interval=arguments.rebalance_interval,
        min_signal=arguments.min_signal,
        target_annual_volatility=arguments.target_volatility,
        market_ma_window=arguments.market_ma_window,
        risk_off_exposure=arguments.risk_off_exposure,
    )
    result = run_oos_portfolio_research(benchmark_name=arguments.benchmark, config=config)
    print(result["message"])
    if result.get("statistics"):
        print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
