"""可复核的单股研究备忘录证据层。

本模块只从本地日线和已提供的量化快照提取事实。它不下载数据、不推断财报或
行业信息；缺少基本面证据时必须明确暴露缺口，避免把技术指标包装成专家结论。
"""

import math

import pandas as pd

from astock_core.strategies.research_profiles import get_research_profile


INSUFFICIENT = "数据不足"
RETURN_WINDOWS = (5, 20, 60, 120, 252)


def _as_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _prepare_history(history_data):
    history = pd.DataFrame(history_data).copy()
    if not {"日期", "收盘"}.issubset(history.columns):
        return None
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    history["收盘"] = pd.to_numeric(history["收盘"], errors="coerce")
    if "成交量" in history:
        history["成交量"] = pd.to_numeric(history["成交量"], errors="coerce")
    history = history.dropna(subset=["日期", "收盘"])
    history = history[history["收盘"] > 0].sort_values("日期").drop_duplicates("日期", keep="last")
    return history.reset_index(drop=True) if not history.empty else None


def build_price_research_evidence(history_data, benchmark_data=None):
    """计算多周期价格、回撤、量能和相对基准事实，不补齐缺失交易日。"""
    history = _prepare_history(history_data)
    if history is None:
        return {"数据状态": "数据不足：需要包含日期、收盘且收盘为正的本地日线。"}

    close = history["收盘"]
    result = {
        "数据状态": "可用",
        "数据截至日期": history.iloc[-1]["日期"].strftime("%Y-%m-%d"),
        "样本交易日数": int(len(history)),
        "最新收盘": round(float(close.iloc[-1]), 4),
        "区间收益率": {},
        "相对沪深300": {},
        "价格位置": {},
        "成交活跃度": INSUFFICIENT,
    }
    for window in RETURN_WINDOWS:
        result["区间收益率"][f"{window}日"] = (
            round((float(close.iloc[-1]) / float(close.iloc[-window - 1]) - 1) * 100, 2)
            if len(close) > window
            else INSUFFICIENT
        )

    position_window = min(len(close), 252)
    window_close = close.tail(position_window)
    high, low = float(window_close.max()), float(window_close.min())
    position = (float(close.iloc[-1]) - low) / (high - low) if high > low else None
    peak = window_close.cummax()
    drawdown = float((window_close / peak - 1).min())
    result["价格位置"] = {
        "观察窗口交易日": int(position_window),
        "窗口最高收盘": round(high, 4),
        "窗口最低收盘": round(low, 4),
        "距窗口高点": round((float(close.iloc[-1]) / high - 1) * 100, 2),
        "窗口最大回撤": round(drawdown * 100, 2),
        "区间位置": round(position * 100, 1) if position is not None else INSUFFICIENT,
    }
    if "成交量" in history and len(history) > 20 and pd.notna(history.iloc[-1]["成交量"]):
        prior_average = history["成交量"].iloc[-21:-1].mean()
        if pd.notna(prior_average) and prior_average > 0:
            result["成交活跃度"] = round(float(history.iloc[-1]["成交量"] / prior_average), 2)

    benchmark = _prepare_history(benchmark_data) if benchmark_data is not None else None
    if benchmark is None:
        result["相对沪深300"] = {f"{window}日": INSUFFICIENT for window in RETURN_WINDOWS}
        return result
    aligned = history[["日期", "收盘"]].merge(
        benchmark[["日期", "收盘"]], on="日期", how="inner", suffixes=("_个股", "_基准")
    ).sort_values("日期")
    for window in RETURN_WINDOWS:
        if len(aligned) > window:
            stock_return = float(aligned["收盘_个股"].iloc[-1] / aligned["收盘_个股"].iloc[-window - 1] - 1)
            benchmark_return = float(aligned["收盘_基准"].iloc[-1] / aligned["收盘_基准"].iloc[-window - 1] - 1)
            result["相对沪深300"][f"{window}日"] = round((stock_return - benchmark_return) * 100, 2)
        else:
            result["相对沪深300"][f"{window}日"] = INSUFFICIENT
    return result


def build_expert_research_memo(
    stock_evidence, price_evidence, fundamental_evidence=None, valuation_evidence=None, peer_comparison=None,
    profile_id=None,
):
    """将可追溯事实组织为论点、反证和验证项，而非给出买卖结论。"""
    current = stock_evidence.get("当前量化证据", {})
    price_evidence = price_evidence or {"数据状态": INSUFFICIENT}
    technical_up = _as_number(current.get("MA5")) is not None and _as_number(current.get("MA20")) is not None and current["MA5"] > current["MA20"]
    relative_20d = price_evidence.get("相对沪深300", {}).get("20日")
    relative_strong = _as_number(relative_20d) is not None and relative_20d > 0
    if technical_up and relative_strong:
        thesis = "技术结构与近20日相对市场表现同向偏强；这只是价格层研究论点，尚不能推导公司基本面改善。"
    elif technical_up:
        thesis = "短期技术结构偏强，但缺少或未确认相对市场优势，论点强度有限。"
    elif relative_strong:
        thesis = "近20日相对市场占优，但短期均线结构未确认，存在信号不一致。"
    else:
        thesis = "现有价格与技术证据未形成相互确认的偏强论点，应以跟踪和证伪为主。"

    supports = []
    returns = price_evidence.get("区间收益率", {})
    if _as_number(returns.get("20日")) is not None:
        supports.append(f"近20个交易日收益为 {returns['20日']}%。")
    if _as_number(relative_20d) is not None:
        supports.append(f"近20个交易日相对沪深300为 {relative_20d:+.2f} 个百分点。")
    if technical_up:
        supports.append("MA5 高于 MA20，短期趋势结构未破坏。")
    if _as_number(current.get("MACD")) is not None:
        supports.append(f"MACD 为 {current['MACD']}。")

    counter = []
    position = price_evidence.get("价格位置", {})
    distance_high = _as_number(position.get("距窗口高点"))
    if distance_high is not None:
        counter.append(f"当前距近{position.get('观察窗口交易日', '观察窗口')}日收盘高点 {distance_high}%；需评估高位或修复阶段的不同风险。")
    rsi = _as_number(current.get("RSI"))
    if rsi is not None and rsi >= 70:
        counter.append(f"RSI 为 {rsi}，位于偏高区间，短线拥挤风险不能忽略。")
    if not technical_up:
        counter.append("MA5 未高于 MA20，短期趋势尚未得到均线确认。")
    if not relative_strong:
        counter.append("近20日未显示相对沪深300优势或数据不足，个股表现可能主要受市场方向驱动。")

    volume_ratio = _as_number(price_evidence.get("成交活跃度"))
    profile = get_research_profile(profile_id)
    confirmation = [
        "下一交易日后继续核对 MA5 与 MA20 的关系及 MACD 状态，确认技术结构没有反转。",
        "核对近20日相对沪深300表现是否保持同方向，避免只在市场普涨时误判个股强势。",
    ]
    if volume_ratio is not None:
        confirmation.append(f"当日成交量为前20日均量的 {volume_ratio} 倍；后续需确认量价是否持续匹配。")
    confirmation.extend(profile["验证项"])

    fundamental_evidence = fundamental_evidence or {}
    fundamental_status = fundamental_evidence.get("数据状态", "数据不足：未下载基本面快照。")
    if fundamental_status == "可用":
        fundamental_text = "；".join(fundamental_evidence.get("事实", []))
        valuation_evidence = valuation_evidence or {}
        if valuation_evidence.get("数据状态") == "可用":
            valuation_facts = []
            if valuation_evidence.get("市净率(PB)") is not None:
                valuation_facts.append(f"PB：{valuation_evidence['市净率(PB)']}。")
            if valuation_evidence.get("静态市盈率(PE)") is not None:
                valuation_facts.append(f"静态PE：{valuation_evidence['静态市盈率(PE)']}。")
            if valuation_facts:
                fundamental_text += "；" + "".join(valuation_facts) + " 估值仅是价格与已披露每股指标的机械比值。"
        peer_comparison = peer_comparison or {}
        if peer_comparison.get("数据状态") == "可用":
            peer_facts = []
            for label, item in peer_comparison.get("指标比较", {}).items():
                if item.get("数据状态") == "可用":
                    peer_facts.append(
                        f"{label}同业排名 {item['同业排名']}/{item['有效可比公司数']}，"
                        f"本公司 {item['本公司']}、同业中位数 {item['同业中位数']}。"
                    )
            if peer_facts:
                fundamental_text += "；同业可比：" + " ".join(peer_facts)
        elif peer_comparison:
            fundamental_text += "；同业可比：" + peer_comparison.get("数据状态", "数据不足")
        fundamental_text += " 请以链接的巨潮资讯官方定期报告复核口径。"
    else:
        fundamental_text = (
            "未接入：尚无经核验的财报、估值、行业景气、竞争格局与管理层指引；"
            "不得以价格信号替代。可显式运行 fundamental_data.py 下载最近报告期快照。"
        )
    return {
        "研究框架": {
            "id": profile["id"],
            "名称": profile["名称"],
            "周期标签": profile["周期标签"],
            "观察周期": profile["观察周期"],
            "研究重点": profile["研究重点"],
            "外部 Skill": profile.get("外部 Skill"),
            "适用边界": profile["适用边界"],
            "来源说明": profile["来源说明"],
        },
        "核心研究论点": thesis,
        "支持证据": supports or ["本地价格证据不足，无法形成支持论据。"],
        "相反证据与风险": counter or ["现有规则未识别到额外反证；不代表风险不存在。"],
        "待验证或证伪": confirmation,
        "基本面与行业证据": fundamental_text,
        "分析边界": "该备忘录仅整合本地日线、量化快照及已保存的基本面事实，不能构成投资建议或公司价值判断。",
    }


def render_expert_research_memo(memo):
    """生成适合报告和页面展示的固定研究结构。"""
    framework = memo.get("研究框架", {})
    lines = ["## 专家研究框架（证据化）", ""]
    if framework:
        lines.extend([
            f"- 当前框架：{framework.get('名称', '未提供')}（{framework.get('周期标签', '未提供')}，{framework.get('观察周期', '未提供')}）",
            f"- 研究重点：{framework.get('研究重点', '未提供')}",
            "",
        ])
    lines.extend([f"- 核心研究论点：{memo['核心研究论点']}", "", "### 支持证据", ""])
    lines.extend(f"- {item}" for item in memo["支持证据"])
    lines.extend(["", "### 相反证据与风险", ""])
    lines.extend(f"- {item}" for item in memo["相反证据与风险"])
    lines.extend(["", "### 待验证或证伪", ""])
    lines.extend(f"- {item}" for item in memo["待验证或证伪"])
    lines.extend(["", f"- 基本面与行业证据：{memo['基本面与行业证据']}", f"- 分析边界：{memo['分析边界']}"])
    return "\n".join(lines)
