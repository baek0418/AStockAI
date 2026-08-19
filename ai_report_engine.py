"""AStockAI v2.9 第二阶段：基于研究摘要生成规则化投研报告。"""

import json
from pathlib import Path

from ai_client import UNAVAILABLE_MESSAGE, call_ai_model


AI_FALLBACK_TEXT = "AI增强分析暂不可用，当前使用规则化分析。"


def find_research_summary_file(output_directory):
    """查找 output 文件夹中最新的研究摘要 JSON 文件。"""
    fixed_summary_file = output_directory / "research_summary.json"

    if fixed_summary_file.exists():
        return fixed_summary_file

    summary_files = sorted(output_directory.glob("research_summary_*.json"))

    if not summary_files:
        raise FileNotFoundError("output 文件夹中没有 research_summary JSON 文件。")

    return summary_files[-1]


def load_research_summary(summary_file):
    """读取研究摘要，并检查生成报告所需的主要字段。"""
    with open(summary_file, "r", encoding="utf-8") as file:
        research_summary = json.load(file)

    required_fields = {
        "快照日期",
        "市场整体状态",
        "评分最高股票TOP3",
        "风险股票列表",
        "回测表现总结",
    }

    if not required_fields.issubset(research_summary):
        raise ValueError("研究摘要缺少生成投研报告所需的字段。")

    return research_summary


def format_number(value):
    """将 JSON 中已有的数值格式化为适合报告展示的两位小数。"""
    return f"{value:.2f}"


def build_ai_fact_snapshot(research_summary):
    """从研究摘要中提取 AI 允许使用的事实快照。"""
    market_status = research_summary.get("市场整体状态", {})
    top_stocks = research_summary.get("评分最高股票TOP3", [])
    risk_stocks = research_summary.get("风险股票列表", [])
    backtest_summary = research_summary.get("回测表现总结", {})

    ai_fact_snapshot = {
        "快照日期": research_summary.get("快照日期", "缺失"),
        "股票池数量": research_summary.get("股票数量", "缺失"),
        "市场整体状态": {
            "状态": market_status.get("状态", "缺失"),
            "平均评分": market_status.get("平均评分", "缺失"),
            "高分股票数量": market_status.get("高分股票数量", "缺失"),
            "风险股票数量": market_status.get("风险股票数量", "缺失"),
        },
        "评分最高股票TOP3": [],
        "风险股票列表": [],
        "回测表现总结": {
            "最终资金": backtest_summary.get("最终资金", "缺失"),
            "累计收益率": backtest_summary.get("累计收益率", "缺失"),
            "胜率": backtest_summary.get("胜率", "缺失"),
            "表现标签": backtest_summary.get("表现标签", "缺失"),
        },
    }

    for stock in top_stocks[:3]:
        ai_fact_snapshot["评分最高股票TOP3"].append(
            {
                "股票名称": stock.get("股票名称", "缺失"),
                "综合评分": stock.get("综合评分", "缺失"),
                "RSI": stock.get("RSI", "缺失"),
                "MA5": stock.get("MA5", "缺失"),
                "MA20": stock.get("MA20", "缺失"),
                "MACD": stock.get("MACD", "缺失"),
                "技术趋势": stock.get("技术趋势", "缺失"),
            }
        )

    for stock in risk_stocks:
        ai_fact_snapshot["风险股票列表"].append(
            {
                "股票名称": stock.get("股票名称", "缺失"),
                "综合评分": stock.get("综合评分", "缺失"),
                "风险原因": stock.get("风险原因", "缺失"),
                "技术趋势": stock.get("技术趋势", "缺失"),
            }
        )

    return ai_fact_snapshot


def create_market_section(market_status):
    """根据研究摘要中的市场状态字段生成市场整体状态章节。"""
    return f"""## 市场整体状态

- 状态：**{market_status['状态']}**
- 股票池平均评分：{format_number(market_status['平均评分'])}
- 高分股票数量：{market_status['高分股票数量']}
- 风险股票数量：{market_status['风险股票数量']}

说明：{market_status['说明']}
"""


def create_top_stocks_section(top_stocks):
    """根据研究摘要中的 TOP3 字段生成高评分股票章节。"""
    lines = [
        "## 股票评分 TOP3",
        "",
        "| 股票 | 综合评分 | 收盘价 | RSI | MA5 | MA20 | MACD | 建议 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for stock in top_stocks:
        lines.append(
            "| "
            f"{stock['股票名称']} | {stock['综合评分']} | "
            f"{format_number(stock['收盘价'])} | {format_number(stock['RSI'])} | "
            f"{format_number(stock['MA5'])} | {format_number(stock['MA20'])} | "
            f"{format_number(stock['MACD'])} | {stock['建议']} |"
        )

    return "\n".join(lines)


def create_risk_stocks_section(risk_stocks):
    """根据研究摘要中的风险股票列表生成风险观察章节。"""
    lines = ["## 风险股票", ""]

    if not risk_stocks:
        lines.append("当前研究摘要中没有风险股票记录。")
        return "\n".join(lines)

    for stock in risk_stocks:
        lines.extend(
            [
                f"### {stock['股票名称']}",
                "",
                f"- 综合评分：{stock['综合评分']}",
                f"- RSI：{format_number(stock['RSI'])}",
                f"- MACD：{format_number(stock['MACD'])}",
                f"- 技术趋势：{stock['技术趋势']}",
                f"- 风险原因：{stock['风险原因']}",
                "",
            ]
        )

    return "\n".join(lines)


def create_technical_trend_section(top_stocks, risk_stocks):
    """使用摘要中已有技术趋势标签生成技术趋势解释章节。"""
    lines = ["## 技术趋势解释", ""]

    for stock in top_stocks:
        lines.append(f"- {stock['股票名称']}：{stock['技术趋势']}。")

    for stock in risk_stocks:
        lines.append(f"- {stock['股票名称']}：{stock['技术趋势']}。")

    return "\n".join(lines)


def create_backtest_section(backtest_summary):
    """根据已有回测表现总结生成策略回测章节。"""
    return f"""## 策略回测表现

- 策略说明：{backtest_summary['策略说明']}
- 回测区间：{backtest_summary['回测开始日期']} 至 {backtest_summary['回测结束日期']}
- 总交易次数：{backtest_summary['总交易次数']}
- 胜率：{format_number(backtest_summary['胜率'])}%（{backtest_summary['胜率标签']}）
- 平均收益率：{format_number(backtest_summary['平均收益率'])}%
- 最终资金：{format_number(backtest_summary['最终资金'])} 元
- 累计收益率：{format_number(backtest_summary['累计收益率'])}%（{backtest_summary['表现标签']}）
- 最大盈利：{format_number(backtest_summary['最大盈利'])}%
- 最大亏损：{format_number(backtest_summary['最大亏损'])}%
"""


def create_rule_based_summary_section(research_summary):
    """基于已有标签生成规则化总结章节。"""
    market_status = research_summary["市场整体状态"]
    top_names = "、".join(
        stock["股票名称"] for stock in research_summary["评分最高股票TOP3"]
    )
    risk_names = "、".join(
        stock["股票名称"] for stock in research_summary["风险股票列表"]
    )
    backtest_summary = research_summary["回测表现总结"]

    if not risk_names:
        risk_names = "当前摘要未列出风险股票"

    return f"""## 规则化总结

当前股票池的量化状态为**{market_status['状态']}**，平均评分为 {format_number(market_status['平均评分'])}。评分靠前的股票为{top_names}，可作为后续研究的优先观察对象。

风险观察对象为{risk_names}，其风险提示和技术趋势已在上文列出，应结合自身风险承受能力审慎评估。

现有策略回测在 {backtest_summary['回测开始日期']} 至 {backtest_summary['回测结束日期']} 的记录中，表现标签为**{backtest_summary['表现标签']}**，胜率标签为**{backtest_summary['胜率标签']}**。本报告仅归纳已有量化事实，不包含新闻、财报、未来价格预测或投资承诺。
"""


def create_ai_style_summary(research_summary):
    """保留旧函数名，兼容早期调用方式。"""
    return create_rule_based_summary_section(research_summary)


def create_risk_disclaimer_section():
    """生成独立的风险声明章节。"""
    return """## 风险声明

- 本报告仅基于历史量化数据和规则化标签生成。
- 报告中的 AI增强分析 只负责解释已有事实，不改变任何评分、排名或回测结果。
- 历史表现不代表未来结果，任何结论都应结合自身风险承受能力独立判断。
- 本报告不构成投资建议，也不保证任何收益。
"""


def build_ai_prompt(research_summary):
    """只使用研究摘要中的已校验事实创建受限的 AI 增强分析提示词。"""
    ai_fact_snapshot = build_ai_fact_snapshot(research_summary)
    summary_json = json.dumps(ai_fact_snapshot, ensure_ascii=False, indent=2)

    return f"""你是量化研究报告助手。请只依据下方提供的研究摘要，输出简洁的中文分析。

必须遵守以下规则：
1. 不允许预测未来涨跌、价格或收益。
2. 不允许编造新闻、财报、政策、行业事件或任何摘要外的信息。
3. 不允许改变、重算或质疑摘要中的评分、收益、排名和技术指标。
4. 只能解释摘要中已经给出的市场状态、技术趋势、风险提示和回测标签。
5. 如果摘要信息不足以支持某项判断，必须明确写“数据不足，无法判断”。
6. 不要给出买卖指令、投资承诺或保证性表述。
7. 不要输出表格，不要重新列出完整数值，只做定性解释。
8. 输出使用中文，语气客观、克制、清晰。
9. 必须说明内容仅用于量化研究和信息展示，不构成投资建议。

研究摘要：
{summary_json}
"""


def generate_ai_enhanced_summary(research_summary):
    """调用大模型生成受限解释，并在服务不可用时返回固定回退文字。"""
    ai_response = call_ai_model(build_ai_prompt(research_summary))

    if ai_response.startswith(UNAVAILABLE_MESSAGE) or not ai_response.strip():
        return AI_FALLBACK_TEXT

    return ai_response


def get_ai_enhanced_analysis(research_summary):
    """保留旧函数名，兼容早期调用方式。"""
    return generate_ai_enhanced_summary(research_summary)


def create_ai_enhanced_section(ai_enhanced_analysis):
    """将 AI 增强分析或固定回退文字放入报告的 AI 总结章节。"""
    return f"""## AI增强分析

{ai_enhanced_analysis}
"""


def create_report_content(research_summary, summary_file, ai_enhanced_analysis):
    """按固定顺序组合全部章节，生成完整 Markdown 报告内容。"""
    top_stocks = research_summary["评分最高股票TOP3"]
    risk_stocks = research_summary["风险股票列表"]

    sections = [
        "# AStockAI 投研报告",
        "",
        f"快照日期：{research_summary['快照日期']}",
        f"研究摘要来源：{summary_file.name}",
        "",
        "> 本报告基于量化事实快照和规则化标签生成，不构成投资建议。",
        "",
        create_market_section(research_summary["市场整体状态"]),
        create_top_stocks_section(top_stocks),
        "",
        create_risk_stocks_section(risk_stocks),
        "",
        create_technical_trend_section(top_stocks, risk_stocks),
        "",
        create_backtest_section(research_summary["回测表现总结"]),
        create_rule_based_summary_section(research_summary),
        create_risk_disclaimer_section(),
        create_ai_enhanced_section(ai_enhanced_analysis),
    ]

    return "\n".join(sections)


def save_report(report_content, snapshot_date, output_directory):
    """将投研报告保存为与研究摘要日期对应的 Markdown 文件。"""
    report_file = output_directory / f"AStockAI投研报告_{snapshot_date}.md"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_content)

    return report_file


def run_ai_report_engine():
    """执行规则化投研报告生成流程，并返回报告内容和保存路径。"""
    output_directory = Path(__file__).parent / "output"
    summary_file = find_research_summary_file(output_directory)
    research_summary = load_research_summary(summary_file)
    ai_enhanced_analysis = generate_ai_enhanced_summary(research_summary)
    report_content = create_report_content(
        research_summary,
        summary_file,
        ai_enhanced_analysis,
    )
    report_file = save_report(
        report_content,
        research_summary["快照日期"],
        output_directory,
    )

    print("投研报告生成成功:")
    print(report_file)
    return report_content, report_file


if __name__ == "__main__":
    run_ai_report_engine()
