"""AStockAI v3.0：根据研究摘要生成关注股票日报。"""

import json
import re
from pathlib import Path

from ai_client import UNAVAILABLE_MESSAGE, call_ai_model
from ai_prompts import RESEARCH_SYSTEM_PROMPT
from analysis_evidence import build_report_evidence


AI_FALLBACK_TEXT = "AI增强分析暂不可用。"
EMAIL_BODY_END_MARKER = "<!-- EMAIL_BODY_END -->"

# 默认关注列表中的代码与名称对应关系，仅用于展示股票名称，不参与指标计算。
STOCK_NAME_MAP = {
    "600519": "贵州茅台",
    "000333": "美的集团",
    "600036": "招商银行",
    "000651": "格力电器",
}


def create_legacy_stock_config(stock_code):
    """将旧版字符串股票代码转换为新版配置字典。"""
    clean_code = str(stock_code).strip()

    return {
        "code": clean_code,
        "name": STOCK_NAME_MAP.get(clean_code, "未知股票"),
        "alias": "",
        "priority": 0,
        "enable": True,
        "tags": [],
        "cost_price": None,
        "target_price": None,
        "notes": "",
    }


def normalize_priority(priority):
    """将配置中的优先级转换为数字，异常值按 0 处理。"""
    if isinstance(priority, bool):
        return 0

    if isinstance(priority, (int, float)):
        return priority

    return 0


def normalize_watchlist_stock(stock_item):
    """整理单只股票配置，并忽略未来新增的未知字段。"""
    if isinstance(stock_item, str):
        return create_legacy_stock_config(stock_item)

    if not isinstance(stock_item, dict):
        return None

    stock_code = str(stock_item.get("code", "")).strip()
    if not stock_code:
        return None

    stock_name = str(stock_item.get("name", "")).strip()
    tags = stock_item.get("tags", [])

    if not isinstance(tags, list):
        tags = []

    return {
        "code": stock_code,
        "name": stock_name or STOCK_NAME_MAP.get(stock_code, "未知股票"),
        "alias": str(stock_item.get("alias", "")).strip(),
        "priority": normalize_priority(stock_item.get("priority", 0)),
        "enable": stock_item.get("enable", True) is not False,
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
        "cost_price": stock_item.get("cost_price"),
        "target_price": stock_item.get("target_price"),
        "notes": str(stock_item.get("notes", "")).strip(),
    }


def create_enabled_watchlist(stock_items):
    """整理关注股票配置，过滤停用项并按优先级稳定排序。"""
    if not isinstance(stock_items, list):
        raise ValueError("watchlist.json 中的 stocks 必须是列表。")

    watchlist_stocks = []
    for stock_item in stock_items:
        normalized_stock = normalize_watchlist_stock(stock_item)
        if normalized_stock and normalized_stock["enable"]:
            watchlist_stocks.append(normalized_stock)

    return sorted(
        watchlist_stocks,
        key=lambda stock: stock["priority"],
        reverse=True,
    )


def load_watchlist(watchlist_file):
    """读取 watchlist.json，并返回已启用且排序后的关注股票配置。"""
    with open(watchlist_file, "r", encoding="utf-8") as file:
        watchlist_data = json.load(file)

    return create_enabled_watchlist(watchlist_data.get("stocks", []))


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
    """读取研究摘要，并检查日报所需的基础字段。"""
    with open(summary_file, "r", encoding="utf-8") as file:
        summary_data = json.load(file)

    required_fields = {"快照日期", "市场整体状态", "评分最高股票TOP3", "风险股票列表"}
    if not required_fields.issubset(summary_data):
        raise ValueError("research_summary 缺少生成关注股票日报所需的字段。")

    return summary_data


def load_daily_signal(output_directory, report_date):
    """优先加载与日报日期一致的日间信号，缺失时保留原有规则日报路径。"""
    signal_file = Path(output_directory) / f"daily_signal_{report_date}.json"
    if not signal_file.is_file():
        return None

    try:
        signal_data = json.loads(signal_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"daily_signal 读取失败：{error}。") from error

    if signal_data.get("快照日期") != report_date or not isinstance(signal_data.get("stocks"), list):
        raise ValueError("daily_signal 日期或 stocks 格式错误。")
    return signal_data


def create_summary_stock_lookup(summary_data):
    """将摘要中的 TOP3 与风险股票按名称整理为便于查询的字典。"""
    stock_lookup = {}

    for stock in summary_data.get("评分最高股票TOP3", []):
        stock_name = stock.get("股票名称")
        if stock_name:
            stock_lookup[stock_name] = dict(stock)

    for stock in summary_data.get("风险股票列表", []):
        stock_name = stock.get("股票名称")
        if not stock_name:
            continue

        existing_stock = stock_lookup.get(stock_name, {})
        stock_lookup[stock_name] = {**existing_stock, **stock}

    return stock_lookup


def get_rsi_label(stock_data):
    """读取摘要中已有的 RSI 标签，缺失时明确标记为未提供。"""
    return stock_data.get("RSI标签", "摘要未提供 RSI 标签")


def get_risk_label(stock_data):
    """读取摘要中已有的风险标签或风险原因，不重新判断风险。"""
    return stock_data.get(
        "风险标签", stock_data.get("风险原因", "摘要未提供风险标签")
    )


def create_watchlist_stocks(watchlist_configs, summary_data):
    """按启用后的关注配置筛选摘要股票，并保留没有数据的提示。"""
    stock_lookup = create_summary_stock_lookup(summary_data)
    watchlist_stocks = []

    for watchlist_config in watchlist_configs:
        stock_code = watchlist_config["code"]
        stock_name = watchlist_config["name"]
        stock_data = stock_lookup.get(stock_name)

        if stock_data is None:
            watchlist_stocks.append(
                {
                    "股票代码": stock_code,
                    "股票名称": stock_name,
                    "摘要数据可用": False,
                    "别名": watchlist_config["alias"],
                    "优先级": watchlist_config["priority"],
                    "标签": watchlist_config["tags"],
                    "持仓成本": watchlist_config["cost_price"],
                    "目标价": watchlist_config["target_price"],
                    "备注": watchlist_config["notes"],
                }
            )
            continue

        watchlist_stocks.append(
            {
                "股票代码": stock_code,
                "股票名称": stock_name,
                "摘要数据可用": True,
                "别名": watchlist_config["alias"],
                "优先级": watchlist_config["priority"],
                "标签": watchlist_config["tags"],
                "持仓成本": watchlist_config["cost_price"],
                "目标价": watchlist_config["target_price"],
                "备注": watchlist_config["notes"],
                "综合评分": stock_data.get("综合评分", "摘要未提供"),
                "RSI": stock_data.get("RSI", "摘要未提供"),
                "技术趋势": stock_data.get("技术趋势", "摘要未提供技术趋势"),
                "RSI标签": get_rsi_label(stock_data),
                "风险标签": get_risk_label(stock_data),
            }
        )

    return watchlist_stocks


def apply_daily_signal(watchlist_stocks, daily_signal):
    """用 daily_signal 的当前事实和变化事实覆盖日报展示字段。"""
    if not daily_signal:
        return watchlist_stocks

    signal_lookup = {}
    for signal_stock in daily_signal["stocks"]:
        if not isinstance(signal_stock, dict):
            continue
        signal_lookup[(signal_stock.get("股票代码"), signal_stock.get("股票名称"))] = signal_stock

    for stock in watchlist_stocks:
        signal_stock = signal_lookup.get((stock["股票代码"], stock["股票名称"]))
        if not signal_stock:
            continue
        current = signal_stock.get("当前指标", {})
        stock["日间信号"] = signal_stock
        stock["信号数据可用"] = True
        stock["综合评分"] = current.get("Score", "数据不足")
        stock["RSI"] = current.get("RSI", "数据不足")
        stock["MA5"] = current.get("MA5", "数据不足")
        stock["MA20"] = current.get("MA20", "数据不足")
        stock["MACD"] = current.get("MACD", "数据不足")
        stock["技术趋势"] = current.get("趋势", "数据不足")
        stock["建议"] = current.get("建议", "数据不足")
        stock["风险标签"] = current.get("风险标签", "数据不足")

    return watchlist_stocks


def build_ai_prompt(summary_data, watchlist_stocks, daily_signal=None):
    """仅使用研究摘要中的事实构造关注股票日报 AI 总结提示词。"""
    ai_facts = {
        "快照日期": summary_data.get("快照日期", "缺失"),
        "市场整体状态": summary_data.get("市场整体状态", {}),
        "关注股票日间信号": daily_signal.get("stocks", []) if daily_signal else [],
        "关注股票摘要事实": [
            {
                "股票名称": stock["股票名称"],
                "综合评分": stock.get("综合评分", "数据不足"),
                "RSI": stock.get("RSI", "数据不足"),
                "技术趋势": stock.get("技术趋势", "数据不足"),
                "风险标签": stock.get("风险标签", "数据不足"),
            }
            for stock in watchlist_stocks
        ],
    }
    facts_text = json.dumps(ai_facts, ensure_ascii=False, indent=2)

    return f"""你是 AStockAI 的量化研究报告助手。

只能解释下面提供的结构化量化事实，尤其是 daily_signal 中的当日变化。
严格使用且只使用以下四个二级标题，不要输出表格：
## 今日整体信号
## 重点变化
## 风险提示
## 下一交易日观察条件
不得补充任何量化事实以外的信息；数据不足时必须明确说明“现有数据不足，无法判断”。

research_summary 事实如下：
{facts_text}
"""


def generate_ai_summary(summary_data, watchlist_stocks, daily_signal=None):
    """调用 AI 解释已有摘要事实，并在服务不可用时返回固定提示。"""
    ai_response = call_ai_model(
        build_ai_prompt(summary_data, watchlist_stocks, daily_signal),
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=800,
    )

    if not isinstance(ai_response, str) or not ai_response.strip():
        return AI_FALLBACK_TEXT

    if ai_response.startswith(UNAVAILABLE_MESSAGE):
        return AI_FALLBACK_TEXT

    return ai_response.strip()


def create_market_section(summary_data):
    """使用研究摘要中已有的市场状态生成报告章节。"""
    market_status = summary_data.get("市场整体状态", {})

    return "\n".join(
        [
            "## 市场整体状态",
            "",
            f"- 状态：{market_status.get('状态', '摘要未提供')}",
            f"- 平均评分：{market_status.get('平均评分', '摘要未提供')}",
            f"- 高分股票数量：{market_status.get('高分股票数量', '摘要未提供')}",
            f"- 风险股票数量：{market_status.get('风险股票数量', '摘要未提供')}",
            f"- 说明：{market_status.get('说明', '摘要未提供说明')}",
        ]
    )


def create_watchlist_section(watchlist_stocks):
    """将关注股票的已有摘要事实写入报告章节。"""
    lines = ["## 关注股票", ""]

    if not watchlist_stocks:
        lines.append("watchlist.json 中没有关注股票代码。")
        return "\n".join(lines)

    for stock in watchlist_stocks:
        lines.extend([f"### {stock['股票名称']}（{stock['股票代码']}）", ""])
        lines.extend(
            [
                f"- 别名：{stock['别名'] or '未设置'}",
                f"- 优先级：{stock['优先级']}",
                f"- 标签：{'、'.join(stock['标签']) or '未设置'}",
            ]
        )

        if stock["备注"]:
            lines.append(f"- 备注：{stock['备注']}")

        if stock["持仓成本"] is not None:
            lines.append(f"- 持仓成本：{stock['持仓成本']}")

        if stock["目标价"] is not None:
            lines.append(f"- 目标价：{stock['目标价']}")

        lines.append("")

        signal_stock = stock.get("日间信号")
        if signal_stock:
            changes = signal_stock["今日变化"]
            lines.extend(
                [
                    f"- Score：{stock['综合评分']}",
                    f"- RSI：{stock['RSI']}",
                    f"- MA5：{stock['MA5']}",
                    f"- MA20：{stock['MA20']}",
                    f"- MACD：{stock['MACD']}",
                    f"- 趋势：{stock['技术趋势']}",
                    f"- 建议：{stock['建议']}",
                    f"- 风险标签：{stock['风险标签']}",
                    f"- 信号分类：{signal_stock['信号分类']}",
                    f"- 数据状态：{signal_stock['数据状态']}",
                    "",
                    "**今日变化**",
                    f"- Score：{changes['Score变化']}",
                    f"- RSI：{changes['RSI变化']}",
                    f"- MA5/MA20：{changes['MA5/MA20关系变化']}",
                    f"- MACD：{changes['MACD状态变化']}",
                    "",
                    "**观察重点**",
                ]
            )
            lines.extend(f"- {condition}" for condition in signal_stock["观察重点"])
            lines.append("")
            continue

        if not stock["摘要数据可用"]:
            lines.extend(
                [
                    "- 当前研究摘要中没有该股票的可用量化记录。",
                    "- 未重新读取 CSV，也未重新计算评分或指标。",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                f"- 综合评分：{stock['综合评分']}",
                f"- RSI：{stock['RSI']}",
                f"- 技术趋势：{stock['技术趋势']}",
                f"- RSI标签：{stock['RSI标签']}",
                f"- 风险标签：{stock['风险标签']}",
                "",
            ]
        )

    return "\n".join(lines)


def create_ai_section(ai_summary):
    """将 AI 总结或固定降级提示写入报告章节。"""
    return f"## AI总结\n\n{ai_summary}"


def create_risk_disclaimer_section():
    """生成固定的风险声明章节。"""
    return """## 风险声明

本报告仅基于已有 research_summary 中的量化事实生成，用于量化研究和信息展示，不构成任何投资建议或收益承诺。历史数据和历史回测不代表未来表现，投资者应独立判断并注意风险。"""


def create_report_content(summary_data, watchlist_stocks, ai_summary):
    """按固定结构组合每日关注股票日报 Markdown 内容。"""
    report_date = summary_data.get("快照日期", "未知日期")

    sections = [
        "# AStockAI 每日关注股票日报",
        "",
        f"日期：{report_date}",
        "",
        "----------------------",
        "",
        create_market_section(summary_data),
        "",
        "----------------------",
        "",
        create_watchlist_section(watchlist_stocks),
        "",
        "----------------------",
        "",
        create_ai_section(ai_summary),
        "",
        "----------------------",
        "",
        create_risk_disclaimer_section(),
        "",
    ]

    return "\n".join(sections)


def save_daily_report(report_content, report_date, output_directory):
    """将日报 Markdown 保存到 output 文件夹。"""
    output_directory.mkdir(exist_ok=True)
    report_file = output_directory / f"每日关注股票日报_{report_date}.md"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_content)

    return report_file


def build_evidence_ai_prompt(evidence):
    """日报 AI 只补充规则摘要未覆盖的观察点，避免重复整份日报。"""
    facts = {
        "报告日期": evidence.get("报告日期"),
        "市场环境": evidence.get("市场环境"),
        "关注股票": [
            {
                "股票代码": stock.get("股票代码"),
                "股票名称": stock.get("股票名称"),
                "优先级": stock.get("优先级"),
                "当前量化证据": stock.get("当前量化证据"),
                "今日变化": stock.get("今日变化"),
                "偏强证据": stock.get("偏强证据"),
                "谨慎证据": stock.get("谨慎证据"),
                "观察重点": stock.get("观察重点"),
            }
            for stock in evidence.get("关注股票", [])
        ],
    }
    return f"""只能解释以下结构化事实，不能添加新闻、公告、资金流、财报或预测概率。
这是一份已经包含市场状态、重点股票和候选提醒的短日报。你的职责是只补充一个最值得留意的矛盾或风险，以及一个下一交易日观察点；不要复述股票清单、市场涨跌、评分或已有结论。
读者不具备技术分析背景：不要直接使用 MACD、RSI、MA5、MA20、均线、金叉、死叉、动能、强弱等术语。请直接说明“近 5 日平均价是否高于近 20 日”“近期价格是否出现持续走强迹象”“近期涨幅是否过大而可能回落”。
严格使用以下两个三级标题，不要输出表格：
### 需要特别留意
### 明日验证点
每个标题下最多两条，总长度不超过 220 个中文字符。每条都必须包含明确对象和至少一项所给事实，并说明该事实与其他信号是否一致；不能只写“出现变化”“等待验证”“持续关注”“谨慎观察”等没有信息量的结论。观察点必须写明具体要核对的信号，例如“近 5 日平均价能否继续高于近 20 日”或“近期价格能否出现持续走强迹象”。不得给出买卖指令、价格预测或收益保证；数据不足时明确说明。
不得给出买卖指令、价格预测、收益保证；数据不足时明确说明。
不得写入事实中未提供的个股收盘价、均线点位、涨跌幅、新闻或原因；
例如事实只有均线关系时，只能写“近 5 日平均价高于/低于近 20 日”，不能补写具体价格或原因。

证据如下：
{json.dumps(facts, ensure_ascii=False, indent=2)}
"""


def generate_evidence_ai_summary(evidence):
    """每日只调用一次 AI，服务不可用时规则日报继续生成。"""
    response = call_ai_model(
        build_evidence_ai_prompt(evidence),
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=360,
    )
    if not isinstance(response, str) or not response.strip():
        return create_rule_cross_signal_summary(evidence, "AI 服务未返回有效内容")
    if response.startswith(UNAVAILABLE_MESSAGE):
        # ai_client 已将认证、配额、网络与响应格式错误转换为不含密钥的
        # 中文说明。报告仍给出可使用的规则解读，避免 AI 降级时出现空洞章节。
        return create_rule_cross_signal_summary(evidence, response)
    return response.strip()


def create_evidence_market_section(market_context):
    """用日常语言展示可追溯的指数变化与趋势背景。"""
    lines = ["## 市场环境", "", f"- 数据截至日期：{market_context.get('数据截至日期', '数据不足')}"]
    for name, item in market_context.get("指数", {}).items():
        if item.get("数据状态") != "可用":
            lines.append(f"- {name}：{item.get('数据状态', '数据不足')}")
            continue
        above = item.get("位于20日均线之上")
        trend_text = (
            "目前高于近 20 个交易日的平均价格，近期整体走势相对偏强。"
            if above is True
            else "目前低于近 20 个交易日的平均价格，近期整体走势仍偏弱。"
            if above is False
            else "近期走势强弱暂无法判断。"
        )
        lines.append(
            f"- {name}：1日 {item.get('1日涨跌')}%，5日 {item.get('5日涨跌')}%，"
            f"20日 {item.get('20日涨跌')}%。{trend_text}"
        )
    return "\n".join(lines)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_report_date(value):
    """将快照序列化后可能携带的时间部分收敛为报告日期。"""
    text = str(value or "未提供").strip()
    return text[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text) else text


def _format_fundamental_metric(label, value):
    """将已披露财务指标写成读者可直接核对的显示口径。"""
    if not _is_number(value):
        return None
    if label in {"营业总收入同比增长", "归母净利润同比增长", "净资产收益率(加权)", "资产负债率"}:
        return f"{label} {value:.2f}%"
    if label == "每股经营现金流":
        return f"{label} {value:.2f} 元/股"
    return f"{label} {value:.2f}"


def describe_score(score):
    """把内部评分说成观察信号，不将评分伪装为预测。"""
    if not _is_number(score):
        return "系统资料不足，暂不作判断"
    if score >= 70:
        label = "系统观察信号相对较好"
    elif score >= 50:
        label = "系统信号相互矛盾"
    else:
        label = "系统观察信号偏弱"
    return f"{label}（观察评分 {score}/100，仅供同一套规则下比较）"


def describe_trend(trend):
    """直接交代近五日和近二十日均价的关系，避免术语标签。"""
    value = str(trend)
    if "多头" in value or "高于" in value:
        return "近 5 日平均价高于近 20 日，近期价格相对走稳"
    if "偏弱" in value or "低于" in value:
        return "近 5 日平均价低于近 20 日，近期价格尚未走强"
    if "数据不足" in value:
        return "走势资料不足"
    return value


def describe_strength(rsi):
    """将 RSI 数值解释为短期价格强弱，保留数值供需要的读者核对。"""
    if not _is_number(rsi):
        return "短期价格强弱资料不足"
    if rsi >= 70:
        label = "近期涨幅较大，需留意回落波动"
    elif rsi <= 30:
        label = "近期卖压较重，波动可能放大"
    elif rsi >= 55:
        label = "短期价格表现偏强"
    elif rsi <= 45:
        label = "短期价格表现偏弱"
    else:
        label = "短期价格强弱较为均衡"
    return f"{label}（强弱值 {rsi}/100）"


def describe_score_change(change):
    if not _is_number(change):
        return "综合状态变化资料不足"
    if change > 0:
        return f"综合状态较上一交易日改善（评分 +{change}）"
    if change < 0:
        return f"综合状态较上一交易日转弱（评分 {change}）"
    return "综合状态与上一交易日基本相同"


def describe_trend_change(change):
    value = str(change)
    if "由MA5 低于 MA20变为MA5 高于 MA20" in value:
        return "近 5 日平均价重新高于近 20 日，近期价格出现走稳迹象"
    if "由MA5 高于 MA20变为MA5 低于 MA20" in value:
        return "近 5 日平均价跌回近 20 日下方，近期价格转弱"
    if "高于 MA20" in value:
        return "近 5 日平均价仍高于近 20 日"
    if "低于 MA20" in value:
        return "近 5 日平均价仍低于近 20 日"
    return "走势变化资料不足" if "数据不足" in value else value


def describe_momentum_change(change):
    """将内部动量状态翻译成近期价格是否呈现走强或转弱迹象。"""
    value = str(change)
    if "转为正值" in value:
        return "近期价格开始出现走强迹象，仍需后续确认"
    if "转为负值" in value or "转为零轴" in value:
        return "此前的走强迹象消失"
    if "正值扩大" in value:
        return "近期走强迹象增强"
    if "正值收窄" in value:
        return "近期仍有走强迹象，但力度减弱"
    if "负值扩大" in value:
        return "近期下行压力加大"
    if "负值收窄" in value:
        return "近期下行压力略有减轻"
    if "维持正值" in value:
        return "近期仍有走强迹象"
    if "维持负值" in value:
        return "近期仍未出现持续走强迹象"
    return "近期价格变化资料不足" if "数据不足" in value else value


def describe_strength_change(change):
    if not _is_number(change):
        return "短期价格强弱变化资料不足"
    if change > 0:
        return f"短期价格表现较前一日改善（强弱值 +{change}）"
    if change < 0:
        return f"短期价格表现较前一日转弱（强弱值 {change}）"
    return "短期价格强弱与前一日基本相同"


def translate_evidence_item(item):
    """将规则层的技术术语改写为报告中的白话解释。"""
    text = str(item)
    replacements = (
        ("MA5 高于 MA20", "近 5 日平均价高于近 20 日平均价"),
        ("MA5 低于或等于 MA20", "近 5 日平均价未高于近 20 日平均价"),
        ("MACD 为正", "近期价格有走强迹象"),
        ("MACD 为负或为零", "近期价格尚未显示持续走强迹象"),
        ("MACD 继续为正", "近期价格仍有走强迹象"),
        ("MACD 继续为负", "近期价格尚未改善"),
        ("MA5 持续高于 MA20", "近 5 日平均价持续高于近 20 日平均价"),
        ("MA5 是否继续高于 MA20", "近 5 日平均价是否持续高于近 20 日平均价"),
        ("MA5 回到 MA20 上方", "近 5 日平均价重新高于近 20 日平均价"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.replace("均线关系", "近 5 日与近 20 日均价的关系")
    text = re.sub(
        r"RSI 为 ([0-9.]+)，位于(.+?)区间",
        r"短期价格强弱为 \1/100，处于\2区间",
        text,
    )
    text = re.sub(
        r"短期价格强弱为 ([0-9.]+)/100，处于偏高区间",
        r"近期上涨较快（\1/100），需留意波动加大",
        text,
    )
    text = re.sub(
        r"短期价格强弱为 ([0-9.]+)/100，处于偏低区间",
        r"近期下跌较多（\1/100），需留意波动加大",
        text,
    )
    text = text.replace("Score", "观察评分")
    return text


def create_reader_summary_section(market_context, stocks):
    """在技术细节前给出无需读指标的一页式结论。"""
    available_indices = [
        item for item in market_context.get("指数", {}).values()
        if item.get("数据状态") == "可用"
    ]
    below_count = sum(item.get("位于20日均线之上") is False for item in available_indices)
    if available_indices and below_count == len(available_indices):
        market_summary = "整体市场偏弱，主要指数都低于近一个月平均价格。"
    elif available_indices and below_count == 0:
        market_summary = "整体市场相对稳定，主要指数都高于近一个月平均价格。"
    else:
        market_summary = "市场强弱分化，不能只看单一指数下结论。"

    matched = [stock for stock in stocks if _is_matched(stock)]
    strongest = sorted(matched, key=_priority, reverse=True)[:3]
    lines = ["## 先看结论", "", f"- 市场：{market_summary}"]
    if strongest:
        lines.append("- 今日重点：" + "；".join(
            f"{stock['股票名称']}—{describe_score(stock['当前量化证据'].get('Score'))}"
            for stock in strongest
        ) + "。")
    else:
        lines.append("- 关注股票的当前资料不足，今天不宜据此作判断。")
    lines.append("- 阅读方式：先看这里和每只股票的“为什么这样看”；后面的指标明细仅供需要时核对。")
    return "\n".join(lines)


def describe_market_condition(market_context):
    """返回日报首页所需的一句话市场环境与对应阅读基调。"""
    available_indices = [
        item for item in market_context.get("指数", {}).values()
        if item.get("数据状态") == "可用"
    ]
    below_count = sum(item.get("位于20日均线之上") is False for item in available_indices)
    if not available_indices:
        return "市场资料不足，今天不宜仅凭个股信号作判断。", "等待资料补全"
    if below_count == len(available_indices):
        return "整体市场偏弱，主要指数都低于近一个月平均价格。", "防守观察，不把近期价格走强直接当成新增机会"
    if below_count == 0:
        return "整体市场相对稳定，主要指数都高于近一个月平均价格。", "可继续跟踪，但仍以个股条件为准"
    return "市场强弱分化，不能只看单一指数下结论。", "聚焦个股变化，避免扩大判断"


def create_daily_brief_section(market_context):
    """邮件正文的入口：先给出结论，再列出可核对的市场事实。"""
    market_summary, stance = describe_market_condition(market_context)
    lines = [
        "## 今日摘要",
        "",
        f"- 市场状态：{market_summary}",
        f"- 今日基调：{stance}。",
    ]
    index_facts = []
    for name, item in market_context.get("指数", {}).items():
        if item.get("数据状态") != "可用":
            continue
        index_facts.append(
            f"{name}：当日 {item.get('1日涨跌')}%，近 5 日 {item.get('5日涨跌')}%，"
            f"近 20 日 {item.get('20日涨跌')}%"
        )
    if index_facts:
        lines.append("- 市场依据：" + "；".join(index_facts) + "。")
    else:
        lines.append("- 市场依据：指数资料不足，未据此扩大个股结论。")
    return "\n".join(lines)


def create_reader_data_status_section(quote_provenance):
    """正文只交代会影响解读边界的数据覆盖与来源状态。"""
    audit = quote_provenance or {}
    total = audit.get("关注股数", 0)
    verified = audit.get("可核对数", 0)
    lines = ["## 数据状态", ""]
    lines.append(
        f"- 日报快照日期：{audit.get('日报快照日期', '数据不足')}；"
        f"关注股行情来源可核对 {verified}/{total}。"
    )
    fallback = audit.get("备用源更新数", 0)
    retries = audit.get("重试后成功数", 0)
    if fallback or retries:
        parts = []
        if fallback:
            parts.append(f"{fallback} 只使用备用源")
        if retries:
            parts.append(f"{retries} 只经重试后成功")
        lines.append("- 更新过程：" + "；".join(parts) + "；数据仍按单一来源前复权口径保存。")
    lines.append("- " + str(audit.get("状态", "数据不足：未找到行情来源审计。")))
    return "\n".join(lines)


def create_quote_provenance_section(quote_provenance):
    """研究附录提供逐股来源核对，不将来源本身解释为市场信号。"""
    audit = quote_provenance or {}
    lines = [
        "## 行情来源与更新核对",
        "",
        "- " + str(audit.get("状态", "数据不足：未找到行情来源审计。")),
        "- 来源记录反映当前本地日线更新情况；若其日期晚于日报快照，不会反向改写本日报结论。",
        "",
        "| 标的 | 行情截至 | 来源与复权 | 更新方式 | 状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in audit.get("股票", []):
        source = item.get("数据源")
        adjustment = item.get("复权方式")
        source_text = f"{source} / {adjustment}" if source and adjustment else "未记录"
        update_text = (
            "未记录"
            if item.get("状态") != "可核对"
            else "备用源" if item.get("是否使用备用源") else "主源"
        )
        attempts = item.get("请求尝试次数")
        if isinstance(attempts, int) and attempts > 1:
            update_text += f"；尝试 {attempts} 次"
        lines.append(
            f"| {item.get('股票名称', '未知股票')}（{item.get('股票代码', '')}） | "
            f"{item.get('数据截至日期', '数据不足')} | {source_text} | {update_text} | "
            f"{item.get('状态', '数据不足')} |"
        )
    if not audit.get("股票"):
        lines.append("| 暂无启用关注股 | 数据不足 | 数据不足 | 数据不足 | 数据不足 |")
    return "\n".join(lines)


def create_research_control_panel_section(evidence):
    """展示日报的资料覆盖、候选闸门和模型准入边界。"""
    market_context = evidence.get("市场环境", {})
    stocks = evidence.get("关注股票", [])
    matched = [stock for stock in stocks if _is_matched(stock)]
    candidate_snapshot = evidence.get("稳健研究候选")
    candidate_market = candidate_snapshot.get("市场环境", {}) if isinstance(candidate_snapshot, dict) else {}
    prediction = evidence.get("预测模型验证", {})
    announcements = evidence.get("公告证据", {})
    quote_provenance = evidence.get("行情来源审计", {})
    if candidate_market.get("passed"):
        candidate_status = f"已开启；当前 {len(candidate_snapshot.get('候选股票', []))} 只通过全部条件"
    else:
        candidate_status = str(candidate_market.get("reason") or "候选资料不足").rstrip("。")
    prediction_status = prediction.get("状态", "未接入")
    prediction_note = str(prediction.get("说明") or "没有可用的预测模型准入结论").rstrip("。")
    lines = [
        "## 研究控制面板",
        "",
        "| 维度 | 今日状态 |",
        "| --- | --- |",
        f"| 数据截止日 | 量化 {evidence.get('报告日期', '数据不足')}；市场 {market_context.get('数据截至日期', '数据不足')} |",
        f"| 行情来源审计 | {quote_provenance.get('可核对数', 0)}/{quote_provenance.get('关注股数', 0)} 只可核对；{quote_provenance.get('状态', '数据不足')} |",
        f"| 关注覆盖 | {len(matched)}/{len(stocks)} 只关注股具备当前量化资料 |",
        f"| 20 日候选闸门 | {candidate_status} |",
        f"| 预测模型准入 | {prediction_status}：{prediction_note} |",
        f"| 官方公告证据 | {announcements.get('status', '数据不足')}；仅展示标题、日期与官方链接 |",
        "",
        "- 本日报优先呈现已验收的规则事实；未通过准入的预测模型不会生成个股结论。",
    ]
    return "\n".join(lines)


def create_official_announcement_section(stock):
    """展示重点股可核对的官方公告事实，不对标题作投资含义推断。"""
    status = stock.get("公告数据状态", "数据不足")
    announcements = stock.get("近期官方公告", [])
    if status != "可用":
        return f"- 官方公告：{status}，今天不据此补充事件解释。"
    if not announcements:
        return "- 官方公告：本次查询未返回近期公告；不代表不存在未收录或盘后披露。"
    facts = "；".join(
        f"{item.get('日期', '日期未知')}《[{item.get('标题', '标题未知')}]({item.get('官方链接', '')})》"
        for item in announcements[:3]
    )
    return "- 官方公告（事实）：" + facts + "。标题分类不代表利好或利空。"


def create_research_priority_board_section(recommendations):
    """把已有的 20 日研究排序放进正文，区分研究优先级与实际候选。"""
    lines = [
        "## 20 日研究跟踪优先级",
        "",
        "- 该列表用于安排研究跟踪顺序，不是买入清单；只有通过市场与个股条件后才会进入“稳健研究候选”。",
        "",
        "| 标的 | 研究优先评分 | 与上一交易日 TOP3 对比 | 当前状态 | 尚未通过的条件 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    if not recommendations:
        lines.append("| 暂无可排序标的 | 数据不足 | 数据不足 | 不据此增加关注 | 数据不足 |")
        return "\n".join(lines)
    for stock in recommendations[:3]:
        gaps = stock.get("未满足条件", [])
        gap_text = "；".join(str(item) for item in gaps) if gaps else "已通过个股条件，仍需市场闸门确认"
        lines.append(
            f"| {stock.get('股票名称', '未知股票')}（{stock.get('股票代码', '')}） | "
            f"{stock.get('20日研究优先评分', '数据不足')} | "
            f"{stock.get('TOP3动态', '上一交易日 TOP3 不可用')} | "
            f"{stock.get('推荐状态', '仅作研究跟踪')} | {gap_text} |"
        )
    return "\n".join(lines)


def _change_impact(stock):
    """按当天变化而非静态评分排序；分数仅用于日报内部筛选。"""
    changes = stock.get("今日变化", {})
    score_change = changes.get("Score变化")
    impact = abs(score_change) * 2 if _is_number(score_change) else 0
    trend_change = str(changes.get("MA5/MA20关系变化", ""))
    momentum_change = str(changes.get("MACD状态变化", ""))
    if "由" in trend_change or "转为" in trend_change:
        impact += 25
    if "由" in momentum_change or "转为" in momentum_change:
        impact += 20
    if "数据不足" in str(stock.get("数据状态", "")):
        impact += 15
    return impact


def select_daily_focus_stocks(stocks, limit=3):
    """最多选出三只真正发生关键变化的股票；无变化时宁可为空。"""
    changed = [stock for stock in stocks if _is_matched(stock) and _has_real_change(stock)]
    return sorted(
        changed,
        key=lambda stock: (_change_impact(stock), _priority(stock)),
        reverse=True,
    )[:limit]


def _is_risk_stock(stock):
    """识别当前仍处于风险状态的关注股，不把单日改善误写成风险解除。"""
    current = stock.get("当前量化证据", {})
    return (
        current.get("建议") == "风险"
        or current.get("风险标签") == "风险"
        or (_is_number(current.get("Score")) and current["Score"] < 50)
    )


def describe_action_status(stock):
    """兼容旧调用的状态概括；首页应优先展示具体量化事实。"""
    if not _is_matched(stock):
        return "资料不足"

    changes = stock.get("今日变化", {})
    current = stock.get("当前量化证据", {})
    has_improvement = (
        _is_number(changes.get("Score变化")) and changes["Score变化"] > 0
    ) or "低于 MA20变为MA5 高于 MA20" in str(changes.get("MA5/MA20关系变化", ""))

    if _is_risk_stock(stock):
        return "风险信号仍在" if has_improvement else "风险状态延续"
    if _has_real_change(stock):
        return "当日信号值得复核"
    if (
        _is_number(current.get("Score"))
        and current["Score"] >= 65
        and "高于" in describe_trend(current.get("趋势"))
    ):
        return "可继续研究跟踪"
    return "常规观察"


def describe_current_momentum(stock):
    """避免展示内部指标名，只说明近期价格是否已有持续走强迹象。"""
    value = stock.get("当前量化证据", {}).get("MACD")
    if not _is_number(value):
        return "近期价格变化资料不足"
    return "近期价格有走强迹象" if value > 0 else "近期价格尚未显示持续走强迹象"


def describe_current_signal(stock):
    """用一行可核对的事实说明当前状态，取代模糊状态标签。"""
    if not _is_matched(stock):
        return "当前量化资料不足"
    current = stock.get("当前量化证据", {})
    score = current.get("Score")
    score_text = f"观察评分 {score}/100" if _is_number(score) else "观察评分资料不足"
    return "；".join([
        score_text,
        describe_trend(current.get("趋势")),
        describe_current_momentum(stock),
    ])


def describe_daily_update(stock):
    """列出当天实际可追溯的变化，不以泛化标签代替事实。"""
    changes = stock.get("今日变化", {})
    pieces = []
    score_change = changes.get("Score变化")
    if _is_number(score_change) and score_change:
        pieces.append(f"观察评分 {'+' if score_change > 0 else ''}{score_change}")
    trend_change = str(changes.get("MA5/MA20关系变化", ""))
    if "由" in trend_change or "转为" in trend_change:
        pieces.append(describe_trend_change(trend_change))
    momentum_change = str(changes.get("MACD状态变化", ""))
    if "转为" in momentum_change or "扩大" in momentum_change or "收窄" in momentum_change:
        pieces.append(describe_momentum_change(momentum_change))
    if pieces:
        return "；".join(pieces)
    if not _is_number(score_change):
        return "缺少与上一交易日的完整对比资料"
    return "未出现需要单独说明的价格状态变化"


def describe_follow_up(stock):
    """给出明确的后续核对项，不直接展示规则层的条件句。"""
    if not _is_matched(stock):
        return "补齐当前量化资料后再判断"
    current = stock.get("当前量化证据", {})
    checks = []
    ma5, ma20 = current.get("MA5"), current.get("MA20")
    if _is_number(ma5) and _is_number(ma20):
        checks.append(
            "近 5 日平均价能否继续高于近 20 日"
            if ma5 > ma20 else "近 5 日平均价能否重新高于近 20 日"
        )
    macd = current.get("MACD")
    if _is_number(macd):
        checks.append("近期价格能否保持走强" if macd > 0 else "近期价格能否出现持续走强迹象")
    rsi = current.get("RSI")
    if _is_number(rsi) and (rsi >= 70 or rsi <= 30):
        checks.append("短期价格强弱是否仍处于极端区间")
    return "；".join(checks[:2]) if checks else "当前指标资料不足，无法设置观察项"


def describe_watchpoint(stock):
    """从当前证据提取最需要留意的事实，避免总览出现模板化条件句。"""
    if not _is_matched(stock):
        return "当前量化资料不足"
    current = stock.get("当前量化证据", {})
    points = []
    score = current.get("Score")
    if _is_number(score) and score < 50:
        points.append(f"观察评分 {score}/100，系统信号偏弱")
    if _is_number(current.get("MA5")) and _is_number(current.get("MA20")) and current["MA5"] <= current["MA20"]:
        points.append("近 5 日平均价低于近 20 日")
    if _is_number(current.get("MACD")) and current["MACD"] <= 0:
        points.append("近期价格尚未显示持续走强迹象")
    rsi = current.get("RSI")
    if _is_number(rsi) and rsi >= 70:
        points.append(f"短期价格强弱 {rsi}/100，处于偏高区间")
    elif _is_number(rsi) and rsi <= 30:
        points.append(f"短期价格强弱 {rsi}/100，处于偏低区间")
    return "；".join(points[:2]) if points else "当前未见突出的规则风险"


def describe_compact_trend_momentum(stock):
    """供全量表横向比较的简短状态，不重复重点区的完整解释。"""
    if not _is_matched(stock):
        return "资料不足"
    current = stock.get("当前量化证据", {})
    trend = current.get("趋势")
    trend_text = (
        "近 5 日均价高于近 20 日" if "高于" in describe_trend(trend)
        else "近 5 日均价低于近 20 日" if "低于" in describe_trend(trend)
        else "价格资料不足"
    )
    momentum = current.get("MACD")
    momentum_text = (
        "有走强迹象" if _is_number(momentum) and momentum > 0
        else "未见持续走强" if _is_number(momentum)
        else "价格变化资料不足"
    )
    return f"{trend_text}；{momentum_text}"


def describe_compact_daily_update(stock):
    """压缩总览的当日信息，只保留评分、趋势或动能的有效变化。"""
    changes = stock.get("今日变化", {})
    pieces = []
    score_change = changes.get("Score变化")
    if _is_number(score_change) and score_change:
        pieces.append(f"评分{'+' if score_change > 0 else ''}{score_change}")

    trend_change = str(changes.get("MA5/MA20关系变化", ""))
    if "由MA5 低于 MA20变为MA5 高于 MA20" in trend_change:
        pieces.append("近期出现走稳迹象")
    elif "由MA5 高于 MA20变为MA5 低于 MA20" in trend_change:
        pieces.append("近期价格转弱")

    momentum_change = str(changes.get("MACD状态变化", ""))
    momentum_labels = (
        ("由负值转为正值", "出现走强迹象"),
        ("由正值转为负值", "近期转弱"),
        ("正值扩大", "走强迹象增强"),
        ("正值收窄", "走强力度减弱"),
        ("负值扩大", "下行压力加大"),
        ("负值收窄", "下行压力减轻"),
    )
    for marker, label in momentum_labels:
        if marker in momentum_change:
            pieces.append(label)
            break
    if pieces:
        return "；".join(pieces)
    return "资料不足" if not _is_number(score_change) else "—"


def describe_compact_watchpoint(stock):
    """总览风险列只显示最重要的一个当前风险，避免一格塞进多句判断。"""
    if not _is_matched(stock):
        return "资料不足"
    current = stock.get("当前量化证据", {})
    score = current.get("Score")
    if _is_number(score) and score < 50:
        return "评分低于 50"
    if _is_number(current.get("MA5")) and _is_number(current.get("MA20")) and current["MA5"] <= current["MA20"]:
        return "近期价格尚未走强"
    if _is_number(current.get("MACD")) and current["MACD"] <= 0:
        return "未见持续走强迹象"
    rsi = current.get("RSI")
    if _is_number(rsi) and rsi >= 70:
        return "短期价格偏高"
    if _is_number(rsi) and rsi <= 30:
        return "短期价格偏低"
    return "—"


def describe_focus_reason(stock):
    """解释为何该股进入今日重点：只引用实际变化及当前信号的一致性。"""
    changes = stock.get("今日变化", {})
    current = stock.get("当前量化证据", {})
    reasons = []
    score_change = changes.get("Score变化")
    if _is_number(score_change) and abs(score_change) >= 10:
        reasons.append(f"观察评分单日{'上调' if score_change > 0 else '下调'} {abs(score_change)} 分")
    raw_trend_change = str(changes.get("MA5/MA20关系变化", ""))
    if "由" in raw_trend_change or "转为" in raw_trend_change:
        reasons.append(describe_trend_change(raw_trend_change))
    raw_momentum_change = str(changes.get("MACD状态变化", ""))
    if "由" in raw_momentum_change or "转为" in raw_momentum_change:
        reasons.append(describe_momentum_change(raw_momentum_change))

    if _is_number(current.get("MA5")) and _is_number(current.get("MA20")) and _is_number(current.get("MACD")):
        if current["MA5"] > current["MA20"] and current["MACD"] <= 0:
            reasons.append("近 5 日均价仍高于近 20 日，但近期价格未出现持续走强迹象，两个信号不一致")
        elif current["MA5"] <= current["MA20"] and current["MACD"] > 0:
            reasons.append("近期价格出现走强迹象，但近 5 日均价仍低于近 20 日，两个信号不一致")
    return "；".join(reasons) if reasons else "当日变化达到重点复核阈值"


def create_rule_cross_signal_summary(evidence, ai_issue=None):
    """AI 不可用时仍输出可读、可追溯的交叉信号解读。"""
    market_context = evidence.get("市场环境", {})
    stocks = [stock for stock in evidence.get("关注股票", []) if _is_matched(stock)]
    market_summary, _ = describe_market_condition(market_context)
    conflicts = []
    for stock in stocks:
        current = stock.get("当前量化证据", {})
        ma5, ma20, macd = current.get("MA5"), current.get("MA20"), current.get("MACD")
        if not all(_is_number(value) for value in (ma5, ma20, macd)):
            continue
        if ma5 > ma20 and macd <= 0:
            conflicts.append(f"{stock['股票名称']}近 5 日均价高于近 20 日，但近期价格尚未显示持续走强迹象")
        elif ma5 <= ma20 and macd > 0:
            conflicts.append(f"{stock['股票名称']}近期价格出现走强迹象，但近 5 日均价仍低于近 20 日")

    focus_stocks = select_daily_focus_stocks(stocks)
    lines = ["### 需要特别留意", "", f"- 市场层面：{market_summary}"]
    if conflicts:
        lines.append("- 个股信号分歧：" + "；".join(conflicts[:2]) + "。")
    elif focus_stocks:
        lines.append("- 当日重点信号：" + "；".join(
            f"{stock['股票名称']}（{describe_daily_update(stock)}）"
            for stock in focus_stocks[:2]
        ) + "。")
    else:
        lines.append("- 个股层面：未出现评分大幅调整或需要单独说明的价格状态变化。")

    lines.extend(["", "### 明日验证点", ""])
    if focus_stocks:
        lines.append("- 重点股：" + "；".join(
            f"{stock['股票名称']}需核对{describe_follow_up(stock)}"
            for stock in focus_stocks[:2]
        ) + "。")
    else:
        lines.append("- 个股：继续核对近 5 日均价与近 20 日均价的关系，以及近期价格是否出现持续走强迹象。")
    available_indices = [
        item for item in market_context.get("指数", {}).values()
        if item.get("数据状态") == "可用"
    ]
    if available_indices:
        above_count = sum(item.get("位于20日均线之上") is True for item in available_indices)
        if above_count == len(available_indices):
            lines.append("- 市场：继续核对主要指数是否仍高于近 20 个交易日平均价格，以判断当前背景能否维持。")
        elif above_count == 0:
            lines.append("- 市场：继续核对主要指数是否仍低于近 20 个交易日平均价格，以判断市场背景是否改善。")
        else:
            lines.append("- 市场：继续核对主要指数的强弱分化是否收敛，避免只凭单一指数扩大判断。")
    if ai_issue:
        lines.append("- 说明：AI增强分析暂不可用，本节已按现有量化事实自动生成。")
    return "\n".join(lines)


def _format_close_price(value):
    if not _is_number(value):
        return "数据不足"
    return f"{value:.2f}"


def create_action_board_section(stocks):
    """日报首页先交代具体事件和风险，而不是罗列抽象状态标签。"""
    matched = [stock for stock in stocks if _is_matched(stock)]
    focus_stocks = select_daily_focus_stocks(stocks)
    changed_stocks = [stock for stock in matched if _has_real_change(stock)]
    risk_stocks = sorted(
        (stock for stock in matched if _is_risk_stock(stock)),
        key=lambda stock: (-_priority(stock), stock["股票名称"]),
    )
    ongoing_stocks = [
        stock for stock in matched
        if stock not in risk_stocks and stock not in focus_stocks
    ]
    lines = ["## 今日重点与风险", ""]
    lines.append(
        f"- 今日有 {len(changed_stocks)} 只关注股出现值得复核的量化信号；"
        f"其中按变化幅度和关注优先级选出 {len(focus_stocks)} 只展开说明；"
        f"当前有 {len(risk_stocks)} 只存在规则风险，其余 {len(ongoing_stocks)} 只按常规节奏跟踪。前两类可重叠。"
    )
    if focus_stocks:
        lines.append("- **今日重点**：")
        lines.extend(
            f"  - {stock['股票名称']}：{describe_focus_reason(stock)}。"
            for stock in focus_stocks
        )
    else:
        lines.append("- **今日重点**：未出现评分大幅调整、趋势切换或动能切换；不因一般日内波动提高个股优先级。")
    if risk_stocks:
        lines.append("- **当前风险**：")
        lines.extend(
            f"  - {stock['股票名称']}：{describe_watchpoint(stock)}。"
            for stock in risk_stocks
        )
    else:
        lines.append("- **当前风险**：当前没有被规则标为风险的关注股票。")
    lines.append("- 以上为量化信号的研究优先级，不构成买卖指令。")
    return "\n".join(lines)


def create_watchlist_overview_section(stocks):
    """以紧凑字段呈现全部关注股，便于在一屏内横向比较。"""
    lines = [
        "## 全量关注速览",
        "",
        "- 此表用于横向比较；详细原因与后续核对项见上方“今日重点”。“—”表示没有需要单独报告的信号，不代表没有价格波动。",
        "",
        "| 股票 | 收盘价 | 观察评分 | 近期价格状态 | 今日变化 | 需要留意 |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    ordered = sorted(stocks, key=lambda stock: (-_priority(stock), stock.get("股票名称", "")))
    for stock in ordered:
        current = stock.get("当前量化证据", {})
        lines.append(
            f"| {stock['股票名称']}（{stock['股票代码']}） | "
            f"{_format_close_price(current.get('收盘价'))} | "
            f"{current.get('Score') if _is_number(current.get('Score')) else '资料不足'} | "
            f"{describe_compact_trend_momentum(stock)} | "
            f"{describe_compact_daily_update(stock)} | "
            f"{describe_compact_watchpoint(stock)} |"
        )
    if not ordered:
        lines.append("| 暂无启用的关注股票 | 数据不足 | 数据不足 | 数据不足 | 数据不足 | 数据不足 |")
    lines.append("")
    lines.append("- 收盘价来自当日量化快照；观察评分（0–100）只用于本系统内的相对排序，不代表涨跌预测。")
    return "\n".join(lines)


def create_daily_focus_section(stocks):
    """把当日事件、当前事实与后续核对项放在同一处，形成可读的重点分析。"""
    focus_stocks = select_daily_focus_stocks(stocks)
    lines = ["## 今日重点", ""]
    if not focus_stocks:
        return "\n".join(lines + ["- 未发现评分大幅调整或需要单独说明的价格状态变化；按既有观察计划跟踪即可。"])

    for stock in focus_stocks:
        current = stock["当前量化证据"]
        changes = stock["今日变化"]
        cautions = stock.get("谨慎证据", [])[:1]
        lines.extend([
            f"### {stock['股票名称']}（{stock['股票代码']}）",
            "",
            f"- 今天出现的变化：{describe_daily_update(stock)}。",
            f"- 现在怎么看：{describe_current_signal(stock)}。",
            f"- 为什么需要留意：{describe_focus_reason(stock)}。",
            "- 下一交易日看什么：" + describe_follow_up(stock) + "。",
            create_official_announcement_section(stock),
        ])
        if cautions and not str(cautions[0]).startswith("未发现"):
            lines.append("- 需要留意：" + translate_evidence_item(cautions[0]))
        lines.append("")
    return "\n".join(lines)


def create_fundamental_review_section(stocks):
    """将重点股的已保存基本面和同业位置写入日报正文，不用缺失数据补结论。"""
    focus_stocks = select_daily_focus_stocks(stocks)
    lines = ["## 重点股基本面与同业核对", ""]
    if not focus_stocks:
        return "\n".join(lines + ["- 今日没有技术层面的重点股，因此不展开基本面核对。"])
    lines.append("- 本节只引用已保存的报告期快照；近期价格变化不等于公司基本面变化。")
    for stock in focus_stocks:
        fundamental = stock.get("基本面研究证据", {})
        peer = stock.get("行业同业比较", {})
        valuation = stock.get("估值观察", {})
        lines.extend([f"### {stock['股票名称']}（{stock['股票代码']}）", ""])
        if fundamental.get("数据状态") != "可用":
            lines.append(f"- 基本面：{fundamental.get('数据状态', '数据不足')}，今天不对公司质量、估值或行业位置作判断。")
            lines.append("")
            continue
        facts = fundamental.get("事实", [])
        report_period = _format_report_date(fundamental.get("报告期"))
        notice_date = _format_report_date(fundamental.get("公告日期"))
        report_fact = f"最新报告期：{report_period}；公告日期：{notice_date}。"
        metrics = fundamental.get("指标", {})
        financial_facts = [
            _format_fundamental_metric(label, metrics.get(label))
            for label in ("营业总收入同比增长", "归母净利润同比增长", "净资产收益率(加权)", "资产负债率", "每股经营现金流")
        ]
        financial_facts = [item for item in financial_facts if item][:3]
        profile_facts = [item for item in facts if item.startswith(("所属行业：", "主营业务："))][:2]
        lines.append(f"- 报告口径：{report_fact}")
        if financial_facts:
            lines.append("- 财务事实：" + "；".join(financial_facts))
        if profile_facts:
            lines.append("- 公司与行业：" + "；".join(profile_facts))
        if valuation.get("数据状态") == "可用":
            pb = valuation.get("市净率(PB)")
            pe = valuation.get("静态市盈率(PE)")
            pb_text = f"{pb:.2f}" if _is_number(pb) else "数据不足"
            pe_text = (
                f"{pe:.2f}"
                if _is_number(pe)
                else "不适用（最新为非年报，静态PE仅在年报口径下展示）"
            )
            lines.append(f"- 估值观察：PB {pb_text}；静态PE {pe_text}。")
        if peer.get("数据状态") == "可用":
            metrics = []
            for label in ("归母净利润同比增长", "净资产收益率(加权)", "资产负债率"):
                item = peer.get("指标比较", {}).get(label, {})
                if item.get("数据状态") == "可用":
                    metrics.append(f"{label}排名 {item['同业排名']}/{item['有效可比公司数']}")
            lines.append(
                f"- 同业位置：{peer.get('所属行业')}，同报告期本地可比 {peer.get('可比公司数量')} 家；"
                + ("；".join(metrics) if metrics else "有效同业指标不足。")
            )
        else:
            lines.append(f"- 同业位置：{peer.get('数据状态', '数据不足')}；不据此比较公司优劣。")
        lines.append("- 核对边界：上述数字须结合巨潮资讯定期报告复核，不构成盈利预测或投资建议。")
        lines.append("")
    return "\n".join(lines)


def create_candidate_alert_section(candidate_snapshot):
    """正文只显示候选是否出现，不重复完整筛选规则。"""
    lines = ["## 候选提醒", ""]
    if not isinstance(candidate_snapshot, dict):
        return "\n".join(lines + ["- 候选清单资料尚未生成，今天不据此增加关注。"])
    market = candidate_snapshot.get("市场环境", {})
    candidates = candidate_snapshot.get("候选股票", [])
    if market.get("passed") and candidates:
        names = "、".join(
            f"{stock.get('股票名称', '未知股票')}（{stock.get('股票代码', '')}）"
            for stock in candidates[:3]
        )
        return "\n".join(lines + [f"- 今日出现 {len(candidates)} 只 20 日稳健研究候选：{names}。详见附录。"])
    reason = str(market.get("reason") or candidate_snapshot.get("说明") or "暂无股票同时满足全部条件").rstrip("。")
    return "\n".join(lines + [f"- 今日无新增 20 日稳健研究候选：{reason}。"])


def create_conservative_candidates_section(candidate_snapshot):
    """展示 20 日稳健研究候选；宁可为空，也不把不足条件的股票凑进来。"""
    lines = ["## 今日 20 日稳健研究候选", ""]
    if not isinstance(candidate_snapshot, dict):
        return "\n".join(lines + [
            "- 当前量化快照尚未生成这份候选清单；下一次完整量化流程后会显示。"
        ])

    market = candidate_snapshot.get("市场环境", {})
    candidates = candidate_snapshot.get("候选股票", [])
    lines.extend([
        "- 这是为约 20 个交易日观察期设计的研究排序，不是买入指令。",
        "- 只有市场环境和个股条件都通过时才会出现候选；系统不会为了凑足三只而放宽门槛。",
    ])
    if not market.get("passed"):
        reason = str(market.get("reason", "市场环境资料不足")).rstrip("。")
        lines.append(f"- 今日无候选：{reason}。")
        return "\n".join(lines)
    if not candidates:
        lines.append(f"- 今日无候选：{candidate_snapshot.get('说明', '没有股票同时满足全部稳健条件')}。")
        return "\n".join(lines)

    for index, stock in enumerate(candidates, start=1):
        lines.extend([
            f"### {index}. {stock.get('股票名称', '未知股票')}（{stock.get('股票代码', '')}）",
            "",
            f"- 为什么进入候选：{'；'.join(stock.get('入选原因', ['资料不足']))}",
            f"- 当前状态：近 20 日涨跌 {stock.get('20日涨跌', '资料不足')}%；"
            f"近 20 日日度波动 {stock.get('20日波动率', '资料不足')}%；"
            f"稳健研究评分 {stock.get('稳健研究评分', '资料不足')}（仅用于候选间排序）。",
            f"- 需要留意：{'；'.join(stock.get('风险提示', ['资料不足']))}",
            "",
        ])
    return "\n".join(lines)


def create_research_recommendations_section(recommendations):
    """展示每日固定的三只 20 日研究优先标的，不将其表述为买卖建议。"""
    lines = ["## 20 日研究优先标的 TOP3", ""]
    lines.append("- 按近期价格表现、近 20 日涨幅、波动与成交活跃度综合排序，用于约 20 个交易日的研究跟踪，不构成买入建议。")
    if not recommendations:
        return "\n".join(lines + ["- 当前量化快照中没有足够的有效股票，暂无法列出三只标的。"])

    for index, stock in enumerate(recommendations, start=1):
        score = stock.get("综合评分")
        rsi = stock.get("RSI")
        trend = "均线多头" if _is_number(stock.get("MA5")) and _is_number(stock.get("MA20")) and stock["MA5"] > stock["MA20"] else "均线偏弱或持平"
        macd = stock.get("MACD")
        gaps = stock.get("未满足条件", [])
        lines.extend([
            f"### {index}. {stock.get('股票名称', '未知股票')}（{stock.get('股票代码', '')}）",
            "",
            f"- 研究优先评分：{stock.get('20日研究优先评分', '数据不足')}/100（仅用于三只标的间排序）。",
            f"- 当前依据：{describe_score(score)}；{describe_trend(trend)}；{describe_strength(rsi)}；"
            f"{'近期价格有走强迹象' if _is_number(macd) and macd > 0 else '近期价格尚未显示持续走强迹象' if _is_number(macd) else '近期价格变化资料不足'}。",
            f"- 当前状态：{stock.get('推荐状态', '仅作研究跟踪')}。",
            f"- 仍待改善：{'；'.join(gaps) if gaps else '个股条件已通过稳健筛选；仍需结合市场环境。'}",
            "",
        ])
    return "\n".join(lines)


def _has_real_change(stock):
    changes = stock.get("今日变化", {})
    score_change = changes.get("Score变化")
    return (
        (isinstance(score_change, (int, float)) and abs(score_change) >= 10)
        or "由" in str(changes.get("MA5/MA20关系变化", ""))
        or "转为" in str(changes.get("MACD状态变化", ""))
        or "数据不足" in str(stock.get("数据状态", ""))
    )


def create_key_changes_section(stocks):
    """只罗列值得关注的变化，并翻译为日常语言。"""
    lines = ["## 今日重点变化", ""]
    changed = [stock for stock in stocks if _has_real_change(stock)]
    if not changed:
        return "\n".join(lines + ["- 未发现满足筛选条件的重点变化；不代表市场没有波动。"])
    for stock in changed:
        changes = stock.get("今日变化", {})
        if not _is_number(changes.get("Score变化")):
            lines.append(
                f"- {stock['股票名称']}：没有可用于比较上一交易日的完整资料，"
                "今天不对它的变化作判断。"
            )
            continue
        lines.append(
            f"- {stock['股票名称']}：{describe_score_change(changes.get('Score变化'))}；"
            f"{describe_trend_change(changes.get('MA5/MA20关系变化'))}；"
            f"{describe_momentum_change(changes.get('MACD状态变化'))}。"
        )
    return "\n".join(lines)


def _priority(stock):
    value = stock.get("优先级")
    return value if isinstance(value, (int, float)) else -1


def _is_matched(stock):
    return stock.get("当前量化证据", {}).get("Score") != "数据不足"


def create_deep_stock_section(stock):
    """以“结论—理由—观察点”的顺序解读重点股票。"""
    current = stock["当前量化证据"]
    changes = stock["今日变化"]
    lines = [f"### {stock['股票名称']}（{stock['股票代码']}）", ""]
    lines.extend([
        f"- 一句话判断：{describe_score(current.get('Score'))}；{describe_trend(current.get('趋势'))}。",
        f"- 短期价格状态：{describe_strength(current.get('RSI'))}。",
        f"- 相比上一交易日：{describe_score_change(changes.get('Score变化'))}；"
        f"{describe_strength_change(changes.get('RSI变化'))}；"
        f"{describe_trend_change(changes.get('MA5/MA20关系变化'))}；"
        f"{describe_momentum_change(changes.get('MACD状态变化'))}。",
        "", "**为什么这样看**",
        *(f"- {translate_evidence_item(item)}" for item in stock.get("偏强证据", ["资料不足"])),
        "", "**需要留意什么**",
        *(f"- {translate_evidence_item(item)}" for item in stock.get("谨慎证据", ["资料不足"])),
        "", "**接下来观察什么**",
        f"- {describe_follow_up(stock)}。", "",
    ])
    return "\n".join(lines)


def create_evidence_watchlist_section(stocks):
    """只对最高优先级三只已匹配股票展开，其余保留简洁事实摘要。"""
    lines = ["## 重点关注股票深度解读", ""]
    matched = sorted((stock for stock in stocks if _is_matched(stock)), key=_priority, reverse=True)
    deep_stocks, other_stocks = matched[:3], matched[3:]
    if not deep_stocks:
        lines.append("- 当前没有已匹配的关注股票，数据不足。")
    for stock in deep_stocks:
        lines.extend([create_deep_stock_section(stock), ""])
    lines.extend(["## 其他关注股票简洁摘要", ""])
    for stock in other_stocks:
        current = stock["当前量化证据"]
        status = str(stock.get("数据状态", "数据不足")).rstrip("。")
        lines.append(
            f"- {stock['股票名称']}：{describe_score(current.get('Score'))}；"
            f"{describe_trend(current.get('趋势'))}；资料状态：{status}。"
        )
    for stock in stocks:
        if not _is_matched(stock):
            lines.append(f"- {stock['股票名称']}：当前量化数据不足，未展开解读。")
    return "\n".join(lines)


def _report_period_label(value):
    """把报告期转为读者易读的年报、中报或季度报告称呼。"""
    report_date = _format_report_date(value)
    if re.fullmatch(r"\d{4}-06-30", report_date):
        return f"{report_date[:4]} 年中报"
    if re.fullmatch(r"\d{4}-12-31", report_date):
        return f"{report_date[:4]} 年报"
    if re.fullmatch(r"\d{4}-03-31", report_date):
        return f"{report_date[:4]} 年一季报"
    if re.fullmatch(r"\d{4}-09-30", report_date):
        return f"{report_date[:4]} 年三季报"
    return f"报告期 {report_date}"


def _reader_price_state(stock):
    """只保留读者需要的当前价格状态，不显示评分或内部指标。"""
    current = stock.get("当前量化证据", {})
    ma5, ma20 = current.get("MA5"), current.get("MA20")
    recent_signal = current.get("MACD")
    if _is_number(ma5) and _is_number(ma20) and ma5 <= ma20:
        return "近期价格仍偏弱"
    if _is_number(ma5) and _is_number(ma20) and ma5 > ma20 and _is_number(recent_signal) and recent_signal > 0:
        return "近期价格相对稳定"
    if _is_number(recent_signal) and recent_signal > 0:
        return "近期开始出现走强迹象，仍待确认"
    return "近期价格资料不足"


def _reader_price_change(stock):
    """只报告实际发生的价格状态切换，避免把评分变化当成分析。"""
    changes = stock.get("今日变化", {})
    parts = []
    trend_change = str(changes.get("MA5/MA20关系变化", ""))
    if "由MA5 低于 MA20变为MA5 高于 MA20" in trend_change:
        parts.append("近 5 日平均价重新高于近 20 日")
    elif "由MA5 高于 MA20变为MA5 低于 MA20" in trend_change:
        parts.append("近 5 日平均价跌回近 20 日下方")
    momentum_change = describe_momentum_change(changes.get("MACD状态变化", ""))
    if momentum_change not in {"近期价格变化资料不足", "MACD 零轴状态未变"} and "资料不足" not in momentum_change:
        parts.append(momentum_change)
    return "；".join(parts[:2]) if parts else "今天未出现需要单独说明的价格状态切换"


def _reader_financial_summary(stock):
    """将最新已披露财务事实组织为一条公司层面的可读信息。"""
    fundamental = stock.get("基本面研究证据", {})
    if fundamental.get("数据状态") != "可用":
        return None
    metrics = fundamental.get("指标", {})
    revenue = metrics.get("营业总收入同比增长")
    profit = metrics.get("归母净利润同比增长")
    roe = metrics.get("净资产收益率(加权)")
    facts = []
    if _is_number(revenue):
        facts.append(f"营收同比{'增长' if revenue >= 0 else '下降'} {abs(revenue):.2f}%")
    if _is_number(profit):
        facts.append(f"归母净利润同比{'增长' if profit >= 0 else '下降'} {abs(profit):.2f}%")
    if _is_number(roe):
        facts.append(f"净资产收益率（ROE）{roe:.2f}%")
    if not facts:
        return None
    return f"最新{_report_period_label(fundamental.get('报告期'))}：" + "，".join(facts) + "。"


def _reader_company_judgment(stock):
    """只根据已披露财务事实和当前价格状态给出边界清晰的分析结论。"""
    fundamental = stock.get("基本面研究证据", {})
    metrics = fundamental.get("指标", {}) if fundamental.get("数据状态") == "可用" else {}
    revenue = metrics.get("营业总收入同比增长")
    profit = metrics.get("归母净利润同比增长")
    price_state = _reader_price_state(stock)
    price_weak = price_state == "近期价格仍偏弱"

    if _is_number(revenue) and _is_number(profit):
        if revenue > 0 and profit > 0 and price_weak:
            return "已披露业绩仍在增长，但价格表现没有同步改善；基本面与价格信号暂不一致。"
        if revenue > 0 and profit <= 0:
            return "收入仍在增长，但利润同比下降；需要先确认盈利能力是否能改善。"
        if revenue <= 0 and profit <= 0 and price_weak:
            return "收入、利润和近期价格表现均偏弱，暂未看到相互印证的改善。"
        if revenue <= 0 and profit <= 0:
            return "收入和利润同比均在下降；即使价格短暂走稳，也不足以说明经营已改善。"
        if revenue > 0 and profit > 0:
            return "收入与利润同比仍在增长；价格表现可继续跟踪，但不能仅凭价格变化外推经营趋势。"
    return "公司基本面资料不足；今天只记录价格变化，不对经营状况作判断。"


def create_reader_market_section(market_context):
    """正文只给出一条市场结论及必要事实，不展示运行面板。"""
    summary, stance = describe_market_condition(market_context)
    facts = []
    for name, item in market_context.get("指数", {}).items():
        if item.get("数据状态") == "可用":
            facts.append(f"{name}近 20 日 {item.get('20日涨跌')}%")
    lines = ["## 今天先看市场", "", f"- {summary}", f"- 今天的处理方式：{stance}。"]
    if facts:
        lines.append("- 参考：" + "；".join(facts) + "。")
    return "\n".join(lines)


def create_reader_focus_section(stocks):
    """正文以公司判断为中心：结论、事实和下一步验证各出现一次。"""
    focus_stocks = select_daily_focus_stocks(stocks)
    lines = ["## 今天最值得关注的公司", ""]
    if not focus_stocks:
        return "\n".join(lines + ["- 今天没有出现需要单独展开的公司变化。"])
    lines.append("- 以下判断只使用已保存的日线和最新报告期快照，不加入新闻或未经核实的原因。")
    for stock in focus_stocks:
        financial = _reader_financial_summary(stock)
        lines.extend([f"### {stock['股票名称']}（{stock['股票代码']}）", ""])
        lines.append(f"- 结论：{_reader_company_judgment(stock)}")
        if financial:
            lines.append(f"- 已披露事实：{financial}")
        else:
            lines.append("- 已披露事实：基本面快照不足，今天不补充公司经营判断。")
        lines.append(f"- 价格变化：{_reader_price_change(stock)}。")
        lines.append(f"- 下一步只看：{describe_follow_up(stock)}。")
        lines.append("")
    lines.append("- 财务数据须以巨潮资讯定期报告复核；这里不构成盈利预测或投资建议。")
    return "\n".join(lines)


def create_reader_watchlist_section(stocks):
    """正文仅保留非重点股的一行式提醒，避免全量信号表打断阅读。"""
    focus_codes = {stock.get("股票代码") for stock in select_daily_focus_stocks(stocks)}
    remaining = [stock for stock in stocks if stock.get("股票代码") not in focus_codes and _is_matched(stock)]
    lines = ["## 其余关注股", ""]
    if not remaining:
        return "\n".join(lines + ["- 其余关注股当前资料不足。"])
    lines.append("- " + "；".join(
        f"{stock['股票名称']}：{_reader_price_state(stock)}"
        for stock in sorted(remaining, key=lambda item: (-_priority(item), item["股票名称"]))
    ) + "。")
    return "\n".join(lines)


def create_evidence_report_content(evidence, ai_summary):
    """组合短日报正文与可按需查看的研究附录。"""
    market_context = evidence.get("市场环境", {})
    stocks = evidence.get("关注股票", [])
    email_sections = [
        "# AStockAI 每日关注股票日报",
        f"日期：{evidence.get('报告日期', '数据不足')}",
        create_reader_market_section(market_context),
        create_reader_data_status_section(evidence.get("行情来源审计")),
        create_reader_focus_section(stocks),
        create_reader_watchlist_section(stocks),
        "> 仅供量化研究参考，不构成投资建议；日线数据不是实时盘中行情。",
    ]
    appendix_sections = [
        "## 研究附录",
        "以下是系统运行状态、全量清单与原始信号，用于核对和回顾，不是每日必读。",
        create_daily_brief_section(market_context),
        create_research_control_panel_section(evidence),
        create_quote_provenance_section(evidence.get("行情来源审计")),
        create_action_board_section(stocks),
        create_candidate_alert_section(evidence.get("稳健研究候选")),
        create_research_priority_board_section(evidence.get("优先研究标的", [])),
        create_daily_focus_section(stocks),
        create_fundamental_review_section(stocks),
        create_watchlist_overview_section(stocks),
        f"## 交叉信号解读\n\n{ai_summary}",
        create_research_recommendations_section(evidence.get("优先研究标的", [])),
        create_conservative_candidates_section(evidence.get("稳健研究候选")),
        create_evidence_market_section(market_context),
        create_key_changes_section(stocks),
        create_evidence_watchlist_section(stocks),
        "## 名词小抄\n\n"
        "- **近 5 日与近 20 日均价**：前者高于后者，只表示近期价格比过去一个月相对稳定，"
        "不代表未来一定上涨。\n"
        "- **近期是否有走强迹象**：系统用已有日线价格判断近期价格表现是否改善；"
        "它只是观察线索，不能单独当作结论。\n"
        "- **近期价格是否偏高或偏低**：上涨或下跌过快都可能带来更大波动，不是买卖信号。\n"
        "- **观察评分**：把已有价格条件汇总为 0–100 的相对参考，只用于阅读排序，不预测收益。",
        "## 风险提示\n\n本报告不构成投资建议；官方公告仅展示标题、日期与链接，未解读公告内容；"
        "未包含新闻、资金流、财报全文或实时盘口；不展示未经验证的预测概率。日线数据不是实时盘中行情。",
    ]
    return "\n\n".join(email_sections + [
        EMAIL_BODY_END_MARKER,
        "<details>",
        "<summary>查看研究附录（静态排序、全量关注列表和术语）</summary>",
        "",
        "\n\n".join(appendix_sections),
        "",
        "</details>",
        "",
    ])


def run_daily_report():
    """执行关注股票日报生成流程，并返回报告路径。"""
    project_directory = Path(__file__).parent
    watchlist_file = project_directory / "watchlist.json"
    output_directory = project_directory / "output"

    evidence = build_report_evidence(output_directory, project_directory / "data" / "market", watchlist_file)
    ai_summary = generate_evidence_ai_summary(evidence)
    report_content = create_evidence_report_content(evidence, ai_summary)
    report_file = save_daily_report(
        report_content, evidence.get("报告日期", "未知日期"), output_directory
    )

    print("每日关注股票日报生成成功:")
    print(report_file)
    return report_file


if __name__ == "__main__":
    run_daily_report()
