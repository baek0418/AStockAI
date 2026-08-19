"""组合级研究回测：样本外信号、调仓约束、交易成本与基准对照。

输入信号必须在对应交易日收盘前已经可得；本模块不会训练模型、补全信号或
读取未来价格。它仅用于研究，不能据此自动交易。
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_SIGNAL_COLUMNS = {"日期", "股票", "收盘", "信号"}
OPTIONAL_TRADE_COLUMNS = {"可买入", "可卖出"}
REBALANCE_COLUMN = "调仓"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PortfolioConfig:
    """组合回测的保守默认参数，费用和交易限制均可显式调整。"""

    initial_capital: float = 100000.0
    max_positions: int = 5
    max_weight: float = 0.20
    rebalance_interval: int = 5
    min_signal: float | None = None
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    sell_stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.001
    lot_size: int = 100

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必须大于 0。")
        if self.max_positions < 1:
            raise ValueError("max_positions 必须至少为 1。")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight 必须在 0 与 1 之间。")
        if self.rebalance_interval < 1:
            raise ValueError("rebalance_interval 必须至少为 1 个交易日。")
        if self.lot_size < 1:
            raise ValueError("lot_size 必须至少为 1。")
        for value, name in (
            (self.commission_rate, "commission_rate"),
            (self.minimum_commission, "minimum_commission"),
            (self.sell_stamp_duty_rate, "sell_stamp_duty_rate"),
            (self.slippage_rate, "slippage_rate"),
        ):
            if value < 0:
                raise ValueError(f"{name} 不能为负数。")


def _as_bool(value, column_name):
    if pd.isna(value):
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "是", "可", "可交易"}:
        return True
    if normalized in {"0", "false", "no", "n", "否", "不可", "停牌", "涨停", "跌停"}:
        return False
    raise ValueError(f"{column_name} 包含无法识别的布尔值：{value}。")


def prepare_signal_panel(signal_data):
    """验证并标准化日期、股票、收盘价、信号与可交易标记。"""
    panel = pd.DataFrame(signal_data).copy()
    missing = REQUIRED_SIGNAL_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"信号数据缺少字段：{sorted(missing)}。")
    if panel.empty:
        raise ValueError("信号数据为空，无法回测。")

    panel["日期"] = pd.to_datetime(panel["日期"], errors="coerce")
    panel["股票"] = panel["股票"].astype(str).str.strip()
    panel["收盘"] = pd.to_numeric(panel["收盘"], errors="coerce")
    panel["信号"] = pd.to_numeric(panel["信号"], errors="coerce")
    if panel[["日期", "股票", "收盘"]].isna().any().any():
        raise ValueError("信号数据含有无效日期、股票或收盘价。")
    if (panel["收盘"] <= 0).any() or (panel["股票"] == "").any():
        raise ValueError("收盘价必须为正数，股票代码/名称不能为空。")
    if panel.duplicated(["日期", "股票"]).any():
        raise ValueError("同一日期、同一股票只能有一条信号。")

    for column in OPTIONAL_TRADE_COLUMNS:
        if column not in panel:
            panel[column] = True
        panel[column] = panel[column].map(lambda value: _as_bool(value, column))
    if REBALANCE_COLUMN in panel:
        panel[REBALANCE_COLUMN] = panel[REBALANCE_COLUMN].map(
            lambda value: _as_bool(value, REBALANCE_COLUMN)
        )
        if panel.groupby("日期")[REBALANCE_COLUMN].nunique().gt(1).any():
            raise ValueError("同一交易日的调仓标记必须对全部股票一致。")
        rebalance_signals = panel.loc[panel[REBALANCE_COLUMN]].groupby("日期")["信号"]
        if rebalance_signals.apply(lambda values: values.notna().any()).eq(False).any():
            raise ValueError("调仓日缺少有效信号。")
    elif panel["信号"].isna().any():
        raise ValueError("未提供调仓标记时，信号不能为空。")
    return panel.sort_values(["日期", "股票"]).reset_index(drop=True)


def prepare_benchmark(benchmark_data, trading_dates):
    """按回测交易日对齐基准；缺失日沿用上一可得收盘价，首日不得缺失。"""
    if benchmark_data is None:
        return None
    benchmark = pd.DataFrame(benchmark_data).copy()
    required = {"日期", "收盘"}
    missing = required - set(benchmark.columns)
    if missing:
        raise ValueError(f"基准数据缺少字段：{sorted(missing)}。")
    benchmark["日期"] = pd.to_datetime(benchmark["日期"], errors="coerce")
    benchmark["收盘"] = pd.to_numeric(benchmark["收盘"], errors="coerce")
    benchmark = benchmark.dropna(subset=["日期", "收盘"]).sort_values("日期")
    if benchmark.empty or (benchmark["收盘"] <= 0).any():
        raise ValueError("基准数据为空或包含非正收盘价。")
    if benchmark.duplicated("日期").any():
        raise ValueError("基准数据的日期不能重复。")
    aligned = benchmark.set_index("日期")["收盘"].reindex(trading_dates).ffill()
    if aligned.isna().any():
        raise ValueError("回测开始日之前缺少基准收盘价。")
    return aligned


def _commission(gross_value, config):
    return max(gross_value * config.commission_rate, config.minimum_commission)


def _buy_cost(shares, close_price, config):
    execution_price = close_price * (1 + config.slippage_rate)
    gross_value = shares * execution_price
    return gross_value + _commission(gross_value, config), execution_price, _commission(gross_value, config)


def _sell_proceeds(shares, close_price, config):
    execution_price = close_price * (1 - config.slippage_rate)
    gross_value = shares * execution_price
    commission = _commission(gross_value, config)
    stamp_duty = gross_value * config.sell_stamp_duty_rate
    return gross_value - commission - stamp_duty, execution_price, commission + stamp_duty


def _affordable_shares(cash, close_price, config):
    lots = int(cash / (close_price * (1 + config.slippage_rate) * config.lot_size))
    while lots > 0:
        shares = lots * config.lot_size
        cost, _, _ = _buy_cost(shares, close_price, config)
        if cost <= cash + 1e-8:
            return shares
        lots -= 1
    return 0


def _position_value(shares, close_price):
    return shares * close_price


def _portfolio_value(cash, positions, today_rows):
    return cash + sum(
        _position_value(shares, float(today_rows.loc[symbol, "收盘"]))
        for symbol, shares in positions.items()
    )


def _select_targets(today_rows, config):
    candidates = today_rows[today_rows["可买入"] & today_rows["信号"].notna()].copy()
    if config.min_signal is not None:
        candidates = candidates[candidates["信号"] >= config.min_signal]
    # ``股票`` 同时是索引和保留列，单独建排序列以兼容 pandas 的歧义检查。
    candidates["_排序股票"] = candidates.index.astype(str)
    candidates = candidates.sort_values(["信号", "_排序股票"], ascending=[False, True])
    return candidates.head(config.max_positions).index.tolist()


def _record_trade(trades, date, symbol, side, shares, price, fee, reason):
    if shares <= 0:
        return
    trades.append(
        {
            "日期": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "股票": symbol,
            "方向": side,
            "股数": int(shares),
            "成交价格": round(float(price), 6),
            "费用": round(float(fee), 6),
            "原因": reason,
        }
    )


def _rebalance(date, today_rows, positions, cash, config, trades):
    """先卖后买，并严格遵守当天的买卖限制和仓位上限。"""
    selected = _select_targets(today_rows, config)
    portfolio_value = _portfolio_value(cash, positions, today_rows)
    target_weight = min(config.max_weight, 1 / config.max_positions)
    target_value = portfolio_value * target_weight

    target_shares = {}
    for symbol in selected:
        row = today_rows.loc[symbol]
        raw_shares = math.floor(target_value / float(row["收盘"]) / config.lot_size)
        target_shares[symbol] = raw_shares * config.lot_size

    for symbol in sorted(list(positions)):
        row = today_rows.loc[symbol]
        desired = target_shares.get(symbol, 0)
        shares = positions[symbol]
        sell_shares = max(shares - desired, 0)
        if sell_shares and not row["可卖出"]:
            continue
        if sell_shares:
            proceeds, execution_price, fee = _sell_proceeds(sell_shares, float(row["收盘"]), config)
            cash += proceeds
            positions[symbol] -= sell_shares
            if positions[symbol] == 0:
                del positions[symbol]
            _record_trade(trades, date, symbol, "卖出", sell_shares, execution_price, fee, "调仓")

    for symbol in selected:
        row = today_rows.loc[symbol]
        if not row["可买入"]:
            continue
        desired = target_shares[symbol]
        held = positions.get(symbol, 0)
        requested = max(desired - held, 0)
        requested -= requested % config.lot_size
        affordable = _affordable_shares(cash, float(row["收盘"]), config)
        buy_shares = min(requested, affordable)
        if buy_shares:
            cost, execution_price, fee = _buy_cost(buy_shares, float(row["收盘"]), config)
            cash -= cost
            positions[symbol] = held + buy_shares
            _record_trade(trades, date, symbol, "买入", buy_shares, execution_price, fee, "调仓")
    return cash


def _maximum_drawdown(net_values):
    running_peak = net_values.cummax()
    drawdowns = net_values / running_peak - 1
    return float(drawdowns.min())


def calculate_performance_statistics(nav_data, trades, config):
    """计算组合、基准和回撤统计；无基准时相关字段为 None。"""
    if nav_data.empty:
        raise ValueError("净值曲线为空，无法计算统计指标。")
    net_values = nav_data["策略净值"].astype(float)
    returns = net_values.pct_change().dropna()
    periods = max(len(net_values) - 1, 1)
    total_return = float(net_values.iloc[-1] / net_values.iloc[0] - 1)
    annualized_return = float((net_values.iloc[-1] / net_values.iloc[0]) ** (TRADING_DAYS_PER_YEAR / periods) - 1)
    volatility = float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) and returns.std(ddof=0) > 0 else None
    statistics = {
        "初始资金": round(float(config.initial_capital), 2),
        "最终资金": round(float(nav_data["组合市值"].iloc[-1]), 2),
        "累计收益率": round(total_return, 6),
        "年化收益率": round(annualized_return, 6),
        "年化波动率": round(volatility, 6),
        "夏普比率": round(sharpe, 6) if sharpe is not None else None,
        "最大回撤": round(_maximum_drawdown(net_values), 6),
        "交易笔数": int(len(trades)),
        "总费用": round(float(pd.DataFrame(trades)["费用"].sum()), 2) if trades else 0.0,
    }
    if "基准净值" in nav_data:
        benchmark_return = float(nav_data["基准净值"].iloc[-1] / nav_data["基准净值"].iloc[0] - 1)
        statistics["基准累计收益率"] = round(benchmark_return, 6)
        statistics["超额累计收益率"] = round(total_return - benchmark_return, 6)
    return statistics


def run_portfolio_backtest(signal_data, config=None, benchmark_data=None):
    """运行组合回测并返回净值曲线、交易记录和统计结果。

    每个交易日只使用当日信号；调用方必须保证该信号是严格样本外、在交易前
    可得的结果。只有提供 ``可买入``/``可卖出`` 为 False 时才会模拟停牌或
    涨跌停限制，模块不会凭空推断限制状态。
    """
    config = config or PortfolioConfig()
    panel = prepare_signal_panel(signal_data)
    trading_dates = pd.Index(panel["日期"].drop_duplicates().sort_values())
    benchmark = prepare_benchmark(benchmark_data, trading_dates)

    cash = float(config.initial_capital)
    positions = {}
    trades = []
    nav_rows = []
    for day_index, date in enumerate(trading_dates):
        today = panel[panel["日期"] == date].copy().set_index("股票", drop=False)
        missing_positions = sorted(set(positions) - set(today.index))
        if missing_positions:
            raise ValueError(f"{date.date()} 缺少持仓行情：{missing_positions}。")
        explicit_rebalance = (
            bool(today[REBALANCE_COLUMN].iloc[0])
            if REBALANCE_COLUMN in today.columns
            else day_index % config.rebalance_interval == 0
        )
        if explicit_rebalance:
            cash = _rebalance(date, today, positions, cash, config, trades)
        portfolio_value = _portfolio_value(cash, positions, today)
        row = {
            "日期": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "现金": round(cash, 6),
            "持仓数量": len(positions),
            "组合市值": round(portfolio_value, 6),
            "策略净值": portfolio_value / config.initial_capital,
        }
        if benchmark is not None:
            row["基准净值"] = float(benchmark.loc[date] / benchmark.iloc[0])
        nav_rows.append(row)

    nav_data = pd.DataFrame(nav_rows)
    trades_data = pd.DataFrame(trades, columns=["日期", "股票", "方向", "股数", "成交价格", "费用", "原因"])
    statistics = calculate_performance_statistics(nav_data, trades, config)
    return nav_data, trades_data, statistics


def save_backtest_report(nav_data, trades_data, statistics, output_directory, config):
    """保存可复核的 JSON、净值与交易明细；输出目录默认被 Git 忽略。"""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().strftime("%Y-%m-%d")
    nav_file = output_path / f"portfolio_nav_{report_date}.csv"
    trades_file = output_path / f"portfolio_trades_{report_date}.csv"
    report_file = output_path / f"portfolio_backtest_{report_date}.json"
    nav_data.to_csv(nav_file, index=False, encoding="utf-8-sig")
    trades_data.to_csv(trades_file, index=False, encoding="utf-8-sig")
    report_file.write_text(
        json.dumps({"参数": asdict(config), "统计": statistics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return nav_file, trades_file, report_file


def main():
    parser = argparse.ArgumentParser(description="AStockAI 组合级研究回测")
    parser.add_argument("signals", help="样本外信号 CSV：日期、股票、收盘、信号，以及可选可买入/可卖出。")
    parser.add_argument("--benchmark", help="可选基准 CSV：日期、收盘。")
    parser.add_argument("--output-dir", default="output/portfolio", help="报告输出目录。")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--min-signal", type=float)
    arguments = parser.parse_args()
    config = PortfolioConfig(
        max_positions=arguments.max_positions,
        rebalance_interval=arguments.rebalance_interval,
        min_signal=arguments.min_signal,
    )
    signal_data = pd.read_csv(arguments.signals)
    benchmark_data = pd.read_csv(arguments.benchmark) if arguments.benchmark else None
    nav_data, trades_data, statistics = run_portfolio_backtest(signal_data, config, benchmark_data)
    _, _, report_file = save_backtest_report(nav_data, trades_data, statistics, arguments.output_dir, config)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"报告：{report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
