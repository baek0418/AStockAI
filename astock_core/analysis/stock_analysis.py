"""AStockAI v3.2：读取已有量化快照生成单股票分析报告。"""

import json
import sys
from pathlib import Path

import pandas as pd

from astock_core.analysis.ai_client import UNAVAILABLE_MESSAGE, call_ai_model
from astock_core.analysis.ai_prompts import RESEARCH_SYSTEM_PROMPT
from astock_core.analysis.analysis_evidence import (
    build_stock_evidence,
    load_latest_daily_signal,
    load_latest_quant_snapshot,
    load_latest_watchlist_snapshot,
    load_market_context,
    load_watchlist_config,
)
from astock_core.analysis.expert_research import (
    build_expert_research_memo,
    build_price_research_evidence,
    render_expert_research_memo,
)
from astock_core.analysis.fundamental_data import (
    build_industry_peer_comparison,
    build_valuation_observation,
    load_fundamental_snapshot,
    summarize_fundamental_evidence,
)


AI_FALLBACK_TEXT = "AI增强分析暂不可用。"


def _read_local_csv(csv_file):
    """读取本地 CSV；分析层只读，不因编码问题改变原始数据。"""
    try:
        return pd.read_csv(csv_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(csv_file)


def enrich_stock_evidence_with_research(stock_evidence, project_directory=None):
    """附加本地价格研究事实和研究备忘录，缺失时保留明确的数据缺口。"""
    project_directory = Path(project_directory or Path(__file__).parents[2])
    enriched = dict(stock_evidence)
    stock_name = str(enriched.get("股票名称", "")).strip()
    history_file = project_directory / "data" / f"{stock_name}历史.csv"
    benchmark_file = project_directory / "data" / "market" / "沪深300_sh000300.csv"
    try:
        history = _read_local_csv(history_file) if history_file.is_file() else None
        benchmark = _read_local_csv(benchmark_file) if benchmark_file.is_file() else None
        price_evidence = build_price_research_evidence(history, benchmark)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        price_evidence = {"数据状态": f"数据不足：本地日线读取失败：{error}。"}
    try:
        fundamental_snapshot = load_fundamental_snapshot(
            enriched.get("股票代码", ""), project_directory / "data" / "fundamentals"
        )
        fundamental_evidence = summarize_fundamental_evidence(fundamental_snapshot)
        peer_comparison = build_industry_peer_comparison(
            fundamental_snapshot, project_directory / "data" / "fundamentals"
        )
    except ValueError as error:
        fundamental_evidence = {"数据状态": f"数据不足：基本面快照股票代码无效：{error}。", "事实": []}
        peer_comparison = {"数据状态": "数据不足：基本面快照股票代码无效，不能进行同业比较。"}
    fundamental_evidence["价格日期"] = price_evidence.get("数据截至日期", "数据不足")
    valuation_evidence = build_valuation_observation(
        fundamental_evidence, price_evidence.get("最新收盘")
    )
    enriched["价格研究证据"] = price_evidence
    enriched["基本面研究证据"] = fundamental_evidence
    enriched["估值观察"] = valuation_evidence
    enriched["行业同业比较"] = peer_comparison
    enriched["专家研究备忘录"] = build_expert_research_memo(
        enriched, price_evidence, fundamental_evidence, valuation_evidence, peer_comparison
    )
    return enriched


def find_latest_snapshot_file(output_directory, file_prefix):
    """查找 output 文件夹中指定类型的最新快照文件。"""
    fixed_file = output_directory / f"{file_prefix}.json"
    if fixed_file.exists():
        return fixed_file

    snapshot_files = sorted(output_directory.glob(f"{file_prefix}_*.json"))
    if not snapshot_files:
        return None

    return snapshot_files[-1]


def load_json_file(json_file, file_description):
    """读取 JSON 文件，并在格式错误时给出明确提示。"""
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"{file_description} JSON 格式错误：{error.msg}。") from error
    except OSError as error:
        raise ValueError(f"无法读取{file_description}：{error}。") from error


def load_quant_snapshot(output_directory):
    """读取最新量化快照，缺失时提示先运行 research_data.py。"""
    snapshot_file = find_latest_snapshot_file(output_directory, "quant_snapshot")
    if snapshot_file is None:
        raise FileNotFoundError("未找到 quant_snapshot，请先运行 research_data.py。")

    quant_snapshot = load_json_file(snapshot_file, "quant_snapshot")
    stock_rankings = quant_snapshot.get("股票排行榜") if isinstance(quant_snapshot, dict) else None
    if not isinstance(stock_rankings, list):
        raise ValueError("quant_snapshot 缺少股票排行榜列表。")

    return quant_snapshot


def load_watchlist_snapshot(output_directory):
    """读取最新关注股票快照；文件不存在时返回 None 以继续分析。"""
    snapshot_file = find_latest_snapshot_file(output_directory, "watchlist_snapshot")
    if snapshot_file is None:
        return None

    watchlist_snapshot = load_json_file(snapshot_file, "watchlist_snapshot")
    stock_list = watchlist_snapshot.get("stocks") if isinstance(watchlist_snapshot, dict) else None
    if not isinstance(stock_list, list):
        raise ValueError("watchlist_snapshot 缺少 stocks 列表。")

    return watchlist_snapshot


def get_stock_code(stock_data):
    """读取中英文兼容的股票代码字段。"""
    for field_name in ("code", "股票代码", "代码"):
        stock_code = stock_data.get(field_name)
        if stock_code is not None and str(stock_code).strip():
            return str(stock_code).strip()

    return ""


def get_stock_name(stock_data):
    """读取中英文兼容的股票名称字段。"""
    for field_name in ("name", "股票名称"):
        stock_name = stock_data.get(field_name)
        if stock_name is not None and str(stock_name).strip():
            return str(stock_name).strip()

    return ""


def create_trend_label(ma5, ma20):
    """根据已有 MA5、MA20 数值生成均线关系描述，不重新计算均线。"""
    if not isinstance(ma5, (int, float)) or not isinstance(ma20, (int, float)):
        return "快照未提供"

    if ma5 > ma20:
        return "MA5 高于 MA20"

    if ma5 < ma20:
        return "MA5 低于 MA20"

    return "MA5 等于 MA20"


def create_quant_record(stock_data):
    """将量化快照中的中文字段转换为统一分析记录。"""
    ma5 = stock_data.get("MA5")
    ma20 = stock_data.get("MA20")

    return {
        "code": get_stock_code(stock_data),
        "name": get_stock_name(stock_data),
        "alias": "",
        "priority": None,
        "tags": [],
        "notes": "",
        "cost_price": None,
        "target_price": None,
        "score": stock_data.get("综合评分"),
        "advice": stock_data.get("建议"),
        "trend": create_trend_label(ma5, ma20),
        "rsi": stock_data.get("RSI"),
        "ma5": ma5,
        "ma20": ma20,
        "macd": stock_data.get("MACD"),
        "risk": "快照未提供",
        "source": "quant_snapshot",
    }


def create_watchlist_record(stock_data):
    """将关注快照中的配置和量化字段转换为统一分析记录。"""
    return {
        "code": get_stock_code(stock_data),
        "name": get_stock_name(stock_data),
        "alias": stock_data.get("alias", ""),
        "priority": stock_data.get("priority"),
        "tags": stock_data.get("tags", []),
        "notes": stock_data.get("notes", ""),
        "cost_price": stock_data.get("cost_price"),
        "target_price": stock_data.get("target_price"),
        "score": stock_data.get("score"),
        "advice": stock_data.get("advice"),
        "trend": stock_data.get("trend", "快照未提供"),
        "rsi": stock_data.get("rsi"),
        "ma5": stock_data.get("ma5"),
        "ma20": stock_data.get("ma20"),
        "macd": stock_data.get("macd"),
        "risk": stock_data.get("risk", "快照未提供"),
        "source": "watchlist_snapshot",
    }


def create_stock_records(quant_snapshot, watchlist_snapshot):
    """合并两类快照记录，并优先使用关注快照中的匹配股票。"""
    stock_records = []
    used_codes = set()
    used_names = set()

    if watchlist_snapshot:
        for stock_data in watchlist_snapshot["stocks"]:
            if not isinstance(stock_data, dict) or stock_data.get("status") == "missing":
                continue

            record = create_watchlist_record(stock_data)
            if not record["code"] and not record["name"]:
                continue

            stock_records.append(record)
            used_codes.add(record["code"])
            used_names.add(record["name"])

    for stock_data in quant_snapshot["股票排行榜"]:
        if not isinstance(stock_data, dict):
            continue

        record = create_quant_record(stock_data)
        if not record["code"] and not record["name"]:
            continue

        if record["code"] in used_codes or record["name"] in used_names:
            continue

        stock_records.append(record)
        used_codes.add(record["code"])
        used_names.add(record["name"])

    return stock_records


def find_fuzzy_matches(stock_query, stock_records):
    """返回名称或别名包含查询文本的全部股票候选项。"""
    query = str(stock_query).strip()
    return [
        stock
        for stock in stock_records
        if query and (query in stock["name"] or query in str(stock["alias"]))
    ]


def find_stock(stock_query, stock_records):
    """按代码、全名、唯一模糊名称或别名匹配股票。"""
    query = str(stock_query).strip()
    if not query:
        return None, "请输入股票代码或名称。"

    code_matches = [stock for stock in stock_records if stock["code"] == query]
    if len(code_matches) == 1:
        return code_matches[0], None

    name_matches = [stock for stock in stock_records if stock["name"] == query]
    if len(name_matches) == 1:
        return name_matches[0], None

    fuzzy_matches = find_fuzzy_matches(query, stock_records)
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], None

    if len(fuzzy_matches) > 1:
        return None, "找到多个股票，请输入完整名称或代码。"

    return None, "未找到该股票。"


def build_fact_snapshot(stock_record):
    """从已加载快照提取单股票报告可用的量化事实。"""
    return {
        "股票代码": stock_record["code"],
        "股票名称": stock_record["name"],
        "别名": stock_record["alias"],
        "优先级": stock_record["priority"],
        "标签": stock_record["tags"],
        "备注": stock_record["notes"],
        "持仓成本": stock_record["cost_price"],
        "目标价": stock_record["target_price"],
        "综合评分": stock_record["score"],
        "建议": stock_record["advice"],
        "趋势": stock_record["trend"],
        "RSI": stock_record["rsi"],
        "MA5": stock_record["ma5"],
        "MA20": stock_record["ma20"],
        "MACD": stock_record["macd"],
        "风险标签": stock_record["risk"],
        "数据来源": stock_record["source"],
    }


def build_rule_summary(fact_snapshot):
    """根据已有量化事实生成不调用 AI 的规则化文字总结。"""
    summary_parts = []
    ma5 = fact_snapshot["MA5"]
    ma20 = fact_snapshot["MA20"]
    rsi = fact_snapshot["RSI"]
    macd = fact_snapshot["MACD"]
    score = fact_snapshot["综合评分"]

    if isinstance(ma5, (int, float)) and isinstance(ma20, (int, float)):
        if ma5 > ma20:
            summary_parts.append("当前 MA5 高于 MA20，均线关系偏强")
        elif ma5 < ma20:
            summary_parts.append("当前 MA5 低于 MA20，均线关系偏弱")
        else:
            summary_parts.append("当前 MA5 与 MA20 持平")

    if isinstance(rsi, (int, float)):
        if rsi >= 70:
            summary_parts.append("RSI 位于偏高区间")
        elif rsi >= 50:
            summary_parts.append("RSI 位于中性偏强区间")
        elif rsi >= 30:
            summary_parts.append("RSI 位于中性偏弱区间")
        else:
            summary_parts.append("RSI 位于偏低区间")

    if isinstance(macd, (int, float)):
        if macd > 0:
            summary_parts.append("MACD 当前为正")
        elif macd < 0:
            summary_parts.append("MACD 当前为负")
        else:
            summary_parts.append("MACD 当前为零")

    if isinstance(score, (int, float)):
        if score >= 80:
            summary_parts.append("综合评分较高")
        elif score >= 65:
            summary_parts.append("综合评分处于较高水平")
        elif score >= 50:
            summary_parts.append("综合评分处于中等水平")
        else:
            summary_parts.append("综合评分偏低")

    if fact_snapshot["建议"]:
        summary_parts.append(f"既有评分建议为“{fact_snapshot['建议']}”")

    if not summary_parts:
        return "现有数据不足，无法生成规则化分析。"

    return "；".join(summary_parts) + "。"


def build_ai_prompt(fact_snapshot):
    """仅使用单股票事实快照构造受限的 AI 总结提示词。"""
    facts_text = json.dumps(fact_snapshot, ensure_ascii=False, indent=2)

    if "当前量化证据" in fact_snapshot:
        return f"""请只解释以下单股票证据包，不得增加任何事实或预测模型概率。
严格使用“市场与趋势背景、近期变化解读、主要风险、条件式观察重点、综合研究结论”五个二级标题；
若证据包包含“专家研究备忘录”，必须把它的支持证据与相反证据都纳入解释，并明确基本面与行业证据是否缺失；
条件只能使用“若……则观察……”表述，不能给出买卖指令或价格预测。

证据包如下：
{facts_text}
"""

    return f"""请仅根据以下单股票结构化量化事实进行简短中文解释。
按“趋势与动量”“风险与观察”“综合结论”三部分组织内容，不要输出表格。
不得重新计算或改变评分、RSI、MA5、MA20、MACD、趋势、建议和风险标签。

量化事实如下：
{facts_text}
"""


def build_ai_summary(fact_snapshot):
    """调用 AI 解释既有事实，并在服务不可用时返回固定提示。"""
    ai_response = call_ai_model(
        build_ai_prompt(fact_snapshot),
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=800,
    )

    if not isinstance(ai_response, str) or not ai_response.strip():
        return AI_FALLBACK_TEXT

    if ai_response.startswith(UNAVAILABLE_MESSAGE):
        return AI_FALLBACK_TEXT

    return ai_response.strip()


def create_stock_information_section(fact_snapshot):
    """生成包含关注配置字段的股票信息 Markdown 章节。"""
    tags = fact_snapshot["标签"]
    lines = ["## 股票信息", ""]
    lines.append(f"- 别名：{fact_snapshot['别名'] or '未设置'}")
    lines.append(f"- 标签：{'、'.join(tags) if tags else '未设置'}")
    lines.append(
        f"- 优先级：{fact_snapshot['优先级'] if fact_snapshot['优先级'] is not None else '未设置'}"
    )

    if fact_snapshot["备注"]:
        lines.append(f"- 备注：{fact_snapshot['备注']}")

    if fact_snapshot["持仓成本"] is not None:
        lines.append(f"- 持仓成本：{fact_snapshot['持仓成本']}")

    if fact_snapshot["目标价"] is not None:
        lines.append(f"- 目标价：{fact_snapshot['目标价']}")

    return "\n".join(lines)


def create_quant_data_section(fact_snapshot):
    """生成当前量化数据 Markdown 章节。"""
    return "\n".join(
        [
            "## 当前量化数据",
            "",
            f"- Score：{fact_snapshot['综合评分']}",
            f"- RSI：{fact_snapshot['RSI']}",
            f"- MA5：{fact_snapshot['MA5']}",
            f"- MA20：{fact_snapshot['MA20']}",
            f"- MACD：{fact_snapshot['MACD']}",
            f"- 趋势：{fact_snapshot['趋势']}",
            f"- 建议：{fact_snapshot['建议']}",
            f"- 风险标签：{fact_snapshot['风险标签']}",
        ]
    )


def create_trend_and_momentum_section(fact_snapshot, rule_summary):
    """输出仅陈述既有 MA、RSI、MACD 事实的趋势与动量章节。"""
    return f"## 趋势与动量\n\n{rule_summary}"


def create_risk_and_observation_section(fact_snapshot):
    """输出风险标签及条件式观察点，不包含预测或确定性买卖指令。"""
    ma5 = fact_snapshot["MA5"]
    ma20 = fact_snapshot["MA20"]
    macd = fact_snapshot["MACD"]
    observations = [f"- 风险标签：{fact_snapshot['风险标签']}"]
    if isinstance(ma5, (int, float)) and isinstance(ma20, (int, float)):
        observations.append(
            "- 观察 MA5 是否继续高于 MA20。"
            if ma5 > ma20
            else "- 观察 MA5 是否继续低于 MA20，或是否回到 MA20 上方。"
        )
    else:
        observations.append("- MA5/MA20 数据不足，无法设置均线观察条件。")

    if isinstance(macd, (int, float)):
        observations.append(
            "- 观察 MACD 是否继续为正。" if macd > 0 else "- 观察 MACD 是否转正。"
        )
    else:
        observations.append("- MACD 数据不足，无法设置动量观察条件。")
    return "## 风险与观察\n\n" + "\n".join(observations)


def create_conclusion_section(fact_snapshot):
    """输出限定在既有量化事实范围内的综合结论。"""
    return (
        "## 综合结论\n\n"
        f"当前综合评分为 {fact_snapshot['综合评分']}，既有评分建议为“{fact_snapshot['建议']}”。"
        "结论仅用于量化研究和信息展示，不构成投资建议。"
    )


def create_markdown_content(fact_snapshot, rule_summary, ai_summary):
    """组合单股票分析报告的完整 Markdown 内容。"""
    return "\n\n".join(
        [
            f"# {fact_snapshot['股票名称']}（{fact_snapshot['股票代码']}）",
            create_stock_information_section(fact_snapshot),
            create_quant_data_section(fact_snapshot),
            create_trend_and_momentum_section(fact_snapshot, rule_summary),
            create_risk_and_observation_section(fact_snapshot),
            create_conclusion_section(fact_snapshot),
            f"## AI增强解释\n\n{ai_summary}",
            "## 风险提示\n\n本报告仅引用已有量化快照中的事实。AI内容仅供参考，不构成投资建议或收益承诺。",
            "",
        ]
    )


def _market_lines(market_context):
    """将证据层市场事实转为 Markdown，不计算或推断指数指标。"""
    lines = ["## 市场环境", "", f"- 市场数据截至日期：{market_context.get('数据截至日期', '数据不足')}" ]
    for name, item in market_context.get("指数", {}).items():
        if item.get("数据状态") != "可用":
            lines.append(f"- {name}：{item.get('数据状态', '数据不足')}")
            continue
        above = item.get("位于20日均线之上")
        above_text = "是" if above is True else "否" if above is False else "数据不足"
        lines.append(
            f"- {name}：数据截至 {item.get('数据截至日期')}；"
            f"1日 {item.get('1日涨跌')}%，5日 {item.get('5日涨跌')}%，20日 {item.get('20日涨跌')}%；"
            f"位于自身20日均线之上：{above_text}。"
        )
    return "\n".join(lines)


def _evidence_list_section(title, items):
    return "\n".join([f"## {title}", "", *(f"- {item}" for item in items)])


def _fundamental_evidence_section(fundamental_evidence, valuation_evidence=None, peer_comparison=None):
    status = fundamental_evidence.get("数据状态", "数据不足")
    lines = ["## 基本面事实快照", "", f"- 数据状态：{status}"]
    lines.extend(f"- {item}" for item in fundamental_evidence.get("事实", []))
    if fundamental_evidence.get("来源"):
        lines.append(f"- 来源：{fundamental_evidence['来源']}")
    if fundamental_evidence.get("官方核验页"):
        lines.append(f"- 官方核验：[巨潮资讯定期报告与财务数据]({fundamental_evidence['官方核验页']})")
    valuation_evidence = valuation_evidence or {}
    if valuation_evidence.get("数据状态") == "可用":
        lines.append(f"- 估值观察：PB {valuation_evidence.get('市净率(PB)', '数据不足')}；静态PE {valuation_evidence.get('静态市盈率(PE)', '不适用/数据不足')}。")
        lines.append(f"- 估值口径：{valuation_evidence.get('说明')}")
    peer_comparison = peer_comparison or {}
    lines.extend(["", "### 行业同业比较", "", f"- 数据状态：{peer_comparison.get('数据状态', '数据不足')}"])
    if peer_comparison.get("数据状态") == "可用":
        lines.append(
            f"- 可比范围：{peer_comparison.get('所属行业')}；报告期 {peer_comparison.get('报告期')}；"
            f"本地可比公司 {peer_comparison.get('可比公司数量')} 家。"
        )
        for label, item in peer_comparison.get("指标比较", {}).items():
            if item.get("数据状态") == "可用":
                lines.append(
                    f"- {label}：本公司 {item['本公司']}；同业中位数 {item['同业中位数']}；"
                    f"排名 {item['同业排名']}/{item['有效可比公司数']}。"
                )
        lines.append(f"- 说明：{peer_comparison.get('说明')}")
    return "\n".join(lines)


def create_evidence_markdown_content(stock_evidence, market_context, ai_summary):
    """组合 v4.6 单股正式报告；所有数值来自证据层的本地文件。"""
    current = stock_evidence["当前量化证据"]
    changes = stock_evidence["今日变化"]
    market_date = market_context.get("数据截至日期", "数据不足")
    sections = [
        f"# {stock_evidence['股票名称']}（{stock_evidence['股票代码']}）",
        "## 数据状态\n\n"
        f"- 个股量化数据截至日期：{stock_evidence['量化数据截至日期']}\n"
        f"- 市场基准数据截至日期：{market_date}\n"
        f"- 个股数据状态：{stock_evidence['数据状态']}\n"
        "- 说明：日线数据不是实时盘中行情。",
        _market_lines(market_context),
        "## 当前量化证据\n\n"
        f"- Score：{current['Score']}\n- 建议：{current['建议']}\n- 趋势：{current['趋势']}\n"
        f"- RSI：{current['RSI']}\n- MA5：{current['MA5']}\n- MA20：{current['MA20']}\n"
        f"- MACD：{current['MACD']}\n- 风险标签：{current['风险标签']}",
        "## 今日变化\n\n"
        f"- Score：{changes.get('Score变化', '数据不足')}\n"
        f"- RSI：{changes.get('RSI变化', '数据不足')}\n"
        f"- MA5/MA20：{changes.get('MA5/MA20关系变化', '数据不足')}\n"
        f"- MACD：{changes.get('MACD状态变化', '数据不足')}",
        _evidence_list_section("偏强证据", stock_evidence["偏强证据"]),
        _evidence_list_section("谨慎证据", stock_evidence["谨慎证据"]),
        _evidence_list_section("条件式观察重点", stock_evidence["观察重点"]),
        _fundamental_evidence_section(
            stock_evidence.get("基本面研究证据", {}), stock_evidence.get("估值观察", {}),
            stock_evidence.get("行业同业比较", {}),
        ),
        render_expert_research_memo(
            stock_evidence.get("专家研究备忘录", build_expert_research_memo(stock_evidence, {}))
        ),
        f"## AI增强分析\n\n{ai_summary}",
        "## 风险提示\n\n"
        "本报告不构成投资建议；未包含新闻或实时盘口；"
        "基本面内容仅限已保存的指标快照，仍须以官方定期报告复核；"
        "不展示未经验证的预测概率。",
        "",
    ]
    return "\n\n".join(sections)


def save_markdown(markdown_content, stock_name, output_directory):
    """将单股票分析报告保存为 Markdown 文件。"""
    safe_name = stock_name.replace("/", "_").replace("\\", "_")
    output_directory.mkdir(exist_ok=True)
    report_file = output_directory / f"{safe_name}_AI分析.md"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    return report_file


def run_stock_analysis(stock_query=None, output_directory=None):
    """执行单股票分析流程，并在异常时输出明确提示。"""
    project_directory = Path(__file__).parents[2]
    target_directory = output_directory or project_directory / "output"

    try:
        quant_snapshot = load_quant_snapshot(target_directory)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return None

    try:
        watchlist_snapshot = load_watchlist_snapshot(target_directory)
    except ValueError as error:
        print(f"watchlist_snapshot 读取失败，将只使用 quant_snapshot：{error}")
        watchlist_snapshot = None

    stock_records = create_stock_records(quant_snapshot, watchlist_snapshot)
    stock_record, error_message = find_stock(stock_query, stock_records)
    if error_message:
        print(error_message)
        return None

    quant_context = load_latest_quant_snapshot(target_directory)
    daily_context = load_latest_daily_signal(target_directory)
    watch_context = load_latest_watchlist_snapshot(target_directory)
    watchlist_context = load_watchlist_config(project_directory / "watchlist.json")
    market_context = load_market_context(project_directory / "data" / "market")
    stock_evidence = build_stock_evidence(
        stock_record, quant_context, daily_context, watch_context, watchlist_context, market_context
    )
    stock_evidence = enrich_stock_evidence_with_research(stock_evidence, project_directory)
    ai_summary = build_ai_summary({**stock_evidence, "市场环境": market_context})
    markdown_content = create_evidence_markdown_content(stock_evidence, market_context, ai_summary)
    report_file = save_markdown(
        markdown_content,
        stock_evidence["股票名称"],
        target_directory,
    )

    print("单股票分析报告生成成功:")
    print(report_file)
    return report_file


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    run_stock_analysis(query)
