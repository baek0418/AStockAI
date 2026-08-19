"""20 日研究优先标的的逐日、无前视历史对照评估。"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from conservative_candidates import (
    HOLDING_DAYS,
    MIN_HISTORY_DAYS,
    _stock_metrics,
    calculate_research_priority_score,
)
from prediction_benchmark_data import BENCHMARKS
from prediction_features import load_history_csv
from score import calculate_score_dataframe
from stock_universe import get_enabled_stock_universe


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
TOP_COUNT = 3


def _future_return(history, as_of_date, holding_days=HOLDING_DAYS):
    positions = history.index[history["日期"] == pd.Timestamp(as_of_date)].tolist()
    if not positions or positions[0] + holding_days >= len(history):
        return None, None, None
    start, end = positions[0], positions[0] + holding_days
    close = history["收盘"].iloc[start : end + 1].astype(float).reset_index(drop=True)
    return round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2), history["日期"].iloc[end].date().isoformat(), close


def _max_drawdown(close):
    curve = close / close.iloc[0]
    return round(float((curve / curve.cummax() - 1).min()) * 100, 2)


def _load_histories(data_directory):
    available_names = {stock["name"]: stock["code"] for stock in get_enabled_stock_universe()}
    histories = {}
    for name, code in available_names.items():
        file_path = Path(data_directory) / f"{name}历史.csv"
        if not file_path.is_file():
            continue
        try:
            histories[name] = {"code": code, "history": load_history_csv(file_path)}
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    return histories


def _candidate_at_date(name, code, history, as_of_date):
    data = history[history["日期"] <= pd.Timestamp(as_of_date)].copy()
    if len(data) < MIN_HISTORY_DAYS or data.iloc[-1]["日期"].date().isoformat() != str(as_of_date):
        return None
    metrics, error = _stock_metrics(history, as_of_date)
    if error:
        return None
    score = calculate_score_dataframe(data)
    priority = calculate_research_priority_score(score["评分"], score["RSI"], score["MACD"], metrics)
    future_return, exit_date, path = _future_return(history, as_of_date)
    if future_return is None:
        return None
    return {
        "股票名称": name,
        "股票代码": code,
        "综合评分": score["评分"],
        "20日研究优先评分": priority,
        "20日收益": future_return,
        "退出日期": exit_date,
        "最大持有期回撤": _max_drawdown(path),
    }


def _strategy_result(candidates, key):
    selected = sorted(candidates, key=lambda item: (-item[key], item["股票名称"]))[:TOP_COUNT]
    if len(selected) < TOP_COUNT:
        return None
    return {
        "股票": [{key: item[key], "股票名称": item["股票名称"], "股票代码": item["股票代码"]} for item in selected],
        "20日组合收益": round(sum(item["20日收益"] for item in selected) / len(selected), 2),
        "最大持有期回撤": round(sum(item["最大持有期回撤"] for item in selected) / len(selected), 2),
    }


def _summary(windows, strategy_key):
    results = [window[strategy_key] for window in windows if window.get(strategy_key)]
    if not results:
        return None
    returns = [item["20日组合收益"] for item in results]
    excess = [item["20日超额收益"] for item in results]
    drawdowns = [item["最大持有期回撤"] for item in results]
    return {
        "样本窗口数": len(results),
        "平均20日收益": round(sum(returns) / len(returns), 2),
        "中位20日收益": round(float(pd.Series(returns).median()), 2),
        "20日收益为正比例": round(sum(value > 0 for value in returns) / len(returns) * 100, 2),
        "平均20日超额收益": round(sum(excess) / len(excess), 2),
        "跑赢沪深300比例": round(sum(value > 0 for value in excess) / len(excess) * 100, 2),
        "平均持有期最大回撤": round(sum(drawdowns) / len(drawdowns), 2),
    }


def evaluate_recommendations(data_directory=None, market_directory=None):
    """逐日构造当时可见的两种 Top3，并比较其后 20 个交易日表现。"""
    data_directory = Path(data_directory or PROJECT_DIRECTORY / "data")
    market_directory = Path(market_directory or data_directory / "market")
    market = load_history_csv(market_directory / BENCHMARKS["沪深300"]["file_name"])
    histories = _load_histories(data_directory)
    windows = []
    for index in range(MIN_HISTORY_DAYS - 1, len(market) - HOLDING_DAYS):
        as_of_date = market.iloc[index]["日期"].date().isoformat()
        market_return, exit_date, _ = _future_return(market, as_of_date)
        if market_return is None:
            continue
        candidates = [
            candidate for name, item in histories.items()
            if (candidate := _candidate_at_date(name, item["code"], item["history"], as_of_date))
        ]
        stable = _strategy_result(candidates, "20日研究优先评分")
        baseline = _strategy_result(candidates, "综合评分")
        for strategy in (stable, baseline):
            if strategy:
                strategy["20日超额收益"] = round(strategy["20日组合收益"] - market_return, 2)
        if stable or baseline:
            windows.append({
                "入选日期": as_of_date,
                "退出日期": exit_date,
                "沪深300 20日收益": market_return,
                "20日研究优先Top3": stable,
                "综合评分Top3": baseline,
            })
    stable_summary = _summary(windows, "20日研究优先Top3")
    baseline_summary = _summary(windows, "综合评分Top3")
    conclusion = "数据不足，尚未形成可比较窗口。"
    if stable_summary and baseline_summary:
        difference = round(stable_summary["平均20日超额收益"] - baseline_summary["平均20日超额收益"], 2)
        hit_rate_difference = round(
            stable_summary["跑赢沪深300比例"] - baseline_summary["跑赢沪深300比例"], 2
        )
        if difference > 0 and hit_rate_difference >= 0:
            conclusion = "20 日研究优先评分在平均超额收益和跑赢基准比例上均不低于综合评分 Top3，可继续扩大样本验证。"
        elif difference > 0:
            conclusion = "20 日研究优先评分的平均超额收益较高，但跑赢基准比例未改善；证据不足以提高其推荐权重，应继续研究。"
        else:
            conclusion = "20 日研究优先评分未优于综合评分 Top3，暂不应继续提高其展示权重。"
    return {
        "研究名称": "20 日研究优先标的滚动评估",
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "持有期（交易日）": HOLDING_DAYS,
        "股票数量": len(histories),
        "窗口": windows,
        "20日研究优先Top3汇总": stable_summary,
        "综合评分Top3汇总": baseline_summary,
        "结论": conclusion,
        "风险提示": [
            "每个窗口仅使用入选日及以前的日线，收益在后 20 个交易日计算。",
            "窗口相互重叠，不能把它当作真实资金曲线或独立样本数。",
            "仅覆盖当前本地股票池；未计交易成本、滑点、停牌和复权口径差异。",
            "结果仅用于研究评估，不构成投资建议或自动交易依据。",
        ],
    }


def create_markdown(report):
    lines = ["# AStockAI 20 日研究优先标的滚动评估", "", f"生成时间：{report['生成时间']}", "", "## 方法", "", "- 每个交易日按当时可见日线选取三只标的，持有 20 个交易日。", "- 对比：20 日研究优先评分 Top3、综合评分 Top3、同期沪深300。", "- 研究优先评分综合趋势、动能、短期价格强弱、20 日涨幅、波动和成交活跃度。", "", "## 汇总结果", ""]
    for title in ("20日研究优先Top3汇总", "综合评分Top3汇总"):
        item = report.get(title)
        lines.append(f"### {title.replace('汇总', '')}")
        if not item:
            lines.extend(["", "- 数据不足。", ""])
            continue
        lines.extend(["", f"- 样本窗口：{item['样本窗口数']}", f"- 平均 20 日收益：{item['平均20日收益']}%", f"- 20 日收益为正比例：{item['20日收益为正比例']}%", f"- 平均 20 日超额收益：{item['平均20日超额收益']}%", f"- 跑赢沪深300比例：{item['跑赢沪深300比例']}%", f"- 平均持有期最大回撤：{item['平均持有期最大回撤']}%", ""])
    lines.extend(["## 结论", "", f"- {report['结论']}", "", "## 风险提示", "", *(f"- {risk}" for risk in report["风险提示"]), ""])
    return "\n".join(lines)


def save_report(report, output_directory=PROJECT_DIRECTORY / "output"):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    json_file = output_directory / f"recommendation_evaluation_{date}.json"
    markdown_file = output_directory / f"recommendation_evaluation_{date}.md"
    json_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_file.write_text(create_markdown(report), encoding="utf-8")
    return json_file, markdown_file


def main():
    report = evaluate_recommendations()
    json_file, markdown_file = save_report(report)
    print(report["结论"])
    print(f"JSON：{json_file}")
    print(f"Markdown：{markdown_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
