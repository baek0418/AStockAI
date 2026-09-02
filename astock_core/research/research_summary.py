"""AStockAI v2.8 第二阶段：将量化事实快照整理为结构化研究摘要。"""

import json
from pathlib import Path


def find_quant_snapshot_file(output_directory):
    """查找 output 文件夹中最新的量化事实快照 JSON 文件。"""
    fixed_snapshot_file = output_directory / "quant_snapshot.json"

    if fixed_snapshot_file.exists():
        return fixed_snapshot_file

    snapshot_files = sorted(output_directory.glob("quant_snapshot_*.json"))

    if not snapshot_files:
        raise FileNotFoundError("output 文件夹中没有 quant_snapshot JSON 文件。")

    return snapshot_files[-1]


def load_quant_snapshot(snapshot_file):
    """读取量化事实快照，并检查生成摘要所需的主要字段。"""
    with open(snapshot_file, "r", encoding="utf-8") as file:
        quant_snapshot = json.load(file)

    required_fields = {"快照日期", "股票排行榜", "策略回测"}

    if not required_fields.issubset(quant_snapshot):
        raise ValueError("量化事实快照缺少生成研究摘要所需的字段。")

    return quant_snapshot


def describe_rsi(rsi):
    """根据已有 RSI 数值生成动量状态标签，不重新计算 RSI。"""
    if rsi >= 70:
        return "RSI偏高"

    if rsi >= 50:
        return "RSI健康偏强"

    if rsi >= 30:
        return "RSI中性偏弱"

    return "RSI偏低"


def describe_technical_trend(stock):
    """根据已有均线、MACD 和 RSI 字段生成技术趋势描述。"""
    if stock["MA5"] > stock["MA20"]:
        ma_description = "均线多头"
    elif stock["MA5"] < stock["MA20"]:
        ma_description = "均线偏弱"
    else:
        ma_description = "均线持平"

    if stock["MACD"] > 0:
        macd_description = "MACD为正"
    elif stock["MACD"] < 0:
        macd_description = "MACD为负"
    else:
        macd_description = "MACD持平"

    return "，".join([ma_description, macd_description, describe_rsi(stock["RSI"])])


def create_top_stocks(stock_rankings):
    """从已排序的股票排行榜中整理评分最高的前三只股票。"""
    top_stocks = []

    for stock in stock_rankings[:3]:
        top_stocks.append(
            {
                "股票名称": stock["股票名称"],
                "综合评分": stock["综合评分"],
                "收盘价": stock["收盘价"],
                "RSI": stock["RSI"],
                "MA5": stock["MA5"],
                "MA20": stock["MA20"],
                "MACD": stock["MACD"],
                "技术趋势": describe_technical_trend(stock),
                "建议": stock["建议"],
            }
        )

    return top_stocks


def create_risk_stocks(stock_rankings):
    """筛选已有建议为风险或综合评分低于 40 分的股票。"""
    risk_stocks = []

    for stock in stock_rankings:
        if stock["建议"] != "风险" and stock["综合评分"] >= 40:
            continue

        risk_stocks.append(
            {
                "股票名称": stock["股票名称"],
                "综合评分": stock["综合评分"],
                "RSI": stock["RSI"],
                "MACD": stock["MACD"],
                "技术趋势": describe_technical_trend(stock),
                "风险原因": "既有评分建议为风险或综合评分低于40分。",
            }
        )

    return risk_stocks


def create_market_status(stock_rankings, risk_stocks):
    """根据股票池已有综合评分生成市场整体状态标签。"""
    average_score = sum(stock["综合评分"] for stock in stock_rankings) / len(
        stock_rankings
    )
    strong_count = sum(stock["综合评分"] >= 70 for stock in stock_rankings)

    if average_score >= 70:
        status = "偏强"
    elif average_score >= 50:
        status = "中性"
    else:
        status = "偏弱"

    return {
        "状态": status,
        "平均评分": round(average_score, 2),
        "高分股票数量": strong_count,
        "风险股票数量": len(risk_stocks),
        "说明": "状态仅依据当前股票池的既有综合评分生成，不代表整个市场指数。",
    }


def create_backtest_summary(backtest_snapshot):
    """读取已有回测统计，生成策略表现标签，不重新执行回测。"""
    statistics = backtest_snapshot["统计"]
    cumulative_return = statistics["累计收益率"]
    win_rate = statistics["胜率"]

    if cumulative_return >= 20:
        performance_label = "策略表现较好"
    elif cumulative_return >= 0:
        performance_label = "策略实现正收益"
    else:
        performance_label = "策略回测亏损"

    if win_rate >= 55:
        win_rate_label = "胜率较高"
    elif win_rate >= 45:
        win_rate_label = "胜率接近均衡"
    else:
        win_rate_label = "胜率偏低"

    return {
        "策略说明": backtest_snapshot["策略说明"],
        "回测开始日期": backtest_snapshot["资金曲线开始日期"],
        "回测结束日期": backtest_snapshot["资金曲线结束日期"],
        "总交易次数": statistics["总交易次数"],
        "胜率": statistics["胜率"],
        "平均收益率": statistics["平均收益率"],
        "最终资金": statistics["最终资金"],
        "累计收益率": cumulative_return,
        "最大盈利": statistics["最大盈利"],
        "最大亏损": statistics["最大亏损"],
        "表现标签": performance_label,
        "胜率标签": win_rate_label,
    }


def create_research_summary(quant_snapshot, snapshot_file):
    """将量化快照转换为供后续 AI 报告使用的结构化研究摘要。"""
    stock_rankings = quant_snapshot["股票排行榜"]
    risk_stocks = create_risk_stocks(stock_rankings)

    return {
        "快照日期": quant_snapshot["快照日期"],
        "量化快照文件": snapshot_file.name,
        "股票数量": quant_snapshot["股票数量"],
        "市场整体状态": create_market_status(stock_rankings, risk_stocks),
        "评分最高股票TOP3": create_top_stocks(stock_rankings),
        "风险股票列表": risk_stocks,
        "回测表现总结": create_backtest_summary(quant_snapshot["策略回测"]),
        "稳健研究候选": quant_snapshot.get("稳健研究候选"),
    }


def save_research_summary(research_summary, output_directory):
    """将结构化研究摘要保存为与量化快照日期对应的 JSON 文件。"""
    summary_file = output_directory / (
        f"research_summary_{research_summary['快照日期']}.json"
    )

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(research_summary, file, ensure_ascii=False, indent=2)

    return summary_file


def run_research_summary():
    """执行研究摘要生成流程，并返回摘要内容和保存路径。"""
    output_directory = Path(__file__).parents[2] / "output"
    snapshot_file = find_quant_snapshot_file(output_directory)
    quant_snapshot = load_quant_snapshot(snapshot_file)
    research_summary = create_research_summary(quant_snapshot, snapshot_file)
    summary_file = save_research_summary(research_summary, output_directory)

    print("研究摘要生成成功:")
    print(summary_file)
    return research_summary, summary_file


if __name__ == "__main__":
    run_research_summary()
