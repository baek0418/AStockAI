"""AStockAI v4.4：本机只读量化查询窗口。"""

import json
from datetime import datetime
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:  # 允许在未安装 Streamlit 的测试环境导入数据辅助函数。
    st = None

from stock_analysis import (
    build_ai_summary,
    build_fact_snapshot,
    build_rule_summary,
    create_markdown_content,
    create_evidence_markdown_content,
    create_stock_records,
    find_fuzzy_matches,
    find_latest_snapshot_file,
    find_stock,
    load_quant_snapshot,
    load_watchlist_snapshot,
)
from analysis_evidence import build_report_evidence, build_stock_evidence
from on_demand_analysis import (
    CATALOG_FILE,
    add_stock_to_watchlist,
    analyze_on_demand_stock,
    load_catalog,
    load_on_demand_snapshot,
    refresh_catalog,
    resolve_code_query,
    resolve_catalog_query,
)


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output"
WATCHLIST_FILE = PROJECT_DIRECTORY / "watchlist.json"


def cache_data(*args, **kwargs):
    """在 Streamlit 环境启用文件缓存；普通 Python 测试时保持函数原样。"""
    if st is None:
        return lambda function: function
    return st.cache_data(*args, **kwargs)


def get_file_mtime_ns(file_path):
    """返回文件纳秒修改时间；缺失文件以 None 表示。"""
    if not file_path:
        return None
    try:
        return Path(file_path).stat().st_mtime_ns
    except OSError:
        return None


def find_latest_file(directory, pattern):
    """按文件名查找日期型输出文件的最新项。"""
    files = sorted(Path(directory).glob(pattern))
    return files[-1] if files else None


def load_json_file(json_file, description):
    """读取本地 JSON；不访问网络，也不读取环境变量。"""
    try:
        return json.loads(Path(json_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取{description}：{error}。") from error


@cache_data(show_spinner=False)
def load_stock_records_cached(output_directory, quant_mtime_ns, watchlist_mtime_ns):
    """按量化与关注快照修改时间缓存股票记录，文件变更后自动失效。"""
    directory = Path(output_directory)
    quant_snapshot = load_quant_snapshot(directory)
    watchlist_snapshot = load_watchlist_snapshot(directory)
    return create_stock_records(quant_snapshot, watchlist_snapshot)


@cache_data(show_spinner=False)
def load_json_cached(json_file, mtime_ns, description):
    """按文件修改时间缓存本地 JSON 内容。"""
    return load_json_file(json_file, description)


@cache_data(show_spinner=False)
def load_text_cached(text_file, mtime_ns):
    """按文件修改时间缓存 Markdown 日报内容。"""
    try:
        return Path(text_file).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"无法读取日报文件：{error}。") from error


@cache_data(show_spinner=False)
def load_catalog_cached(catalog_file, mtime_ns):
    """按目录缓存文件修改时间加载本地 A 股名称目录。"""
    return load_catalog(catalog_file)


@cache_data(show_spinner=False)
def load_report_evidence_cached(output_directory, quant_mtime_ns, signal_mtime_ns, watch_mtime_ns, market_mtime_ns, watchlist_mtime_ns):
    """按快照、市场 CSV 与配置修改时间缓存只读证据包。"""
    directory = Path(output_directory)
    return build_report_evidence(directory, PROJECT_DIRECTORY / "data" / "market", WATCHLIST_FILE)


def get_stock_evidence(stock_record):
    """仅读取本地证据包，为页面补充市场与多空事实。"""
    quant_file = find_latest_file(OUTPUT_DIRECTORY, "quant_snapshot_*.json")
    signal_file = find_latest_file(OUTPUT_DIRECTORY, "daily_signal_*.json")
    watch_file = find_latest_file(OUTPUT_DIRECTORY, "watchlist_snapshot_*.json")
    market_mtime = max(
        get_file_mtime_ns(PROJECT_DIRECTORY / "data" / "market" / "沪深300_sh000300.csv") or 0,
        get_file_mtime_ns(PROJECT_DIRECTORY / "data" / "market" / "中证1000_sh000852.csv") or 0,
    )
    evidence = load_report_evidence_cached(
        str(OUTPUT_DIRECTORY), get_file_mtime_ns(quant_file), get_file_mtime_ns(signal_file),
        get_file_mtime_ns(watch_file), market_mtime, get_file_mtime_ns(WATCHLIST_FILE),
    )
    contexts = evidence["量化快照"], evidence["daily_signal"], evidence["watchlist_snapshot"]
    return build_stock_evidence(
        stock_record, *contexts,
        {"available": True, "stocks": []}, evidence["市场环境"],
    ), evidence["市场环境"]


def get_signal_stock(daily_signal, stock_record):
    """从当天 daily_signal 中按代码优先、名称后备查找变化事实。"""
    if not isinstance(daily_signal, dict):
        return None
    stock_code = stock_record.get("code")
    stock_name = stock_record.get("name")
    for stock in daily_signal.get("stocks", []):
        if not isinstance(stock, dict):
            continue
        if stock.get("股票代码") == stock_code or stock.get("股票名称") == stock_name:
            return stock
    return None


def get_watchlist_rows(watchlist_data):
    """整理只读关注列表展示行。"""
    rows = []
    for stock in watchlist_data.get("stocks", []) if isinstance(watchlist_data, dict) else []:
        if not isinstance(stock, dict):
            continue
        rows.append(
            {
                "代码": stock.get("code", ""),
                "名称": stock.get("name", ""),
                "别名": stock.get("alias", ""),
                "优先级": stock.get("priority", ""),
                "启用": stock.get("enable", True),
                "标签": "、".join(stock.get("tags", [])),
                "备注": stock.get("notes", ""),
            }
        )
    return rows


def get_report_file(report_date):
    """优先返回与最新量化日期对应的日报，缺失时返回已有最新日报。"""
    dated_file = OUTPUT_DIRECTORY / f"每日关注股票日报_{report_date}.md"
    return dated_file if dated_file.is_file() else find_latest_file(OUTPUT_DIRECTORY, "每日关注股票日报_*.md")


def render_home(quant_snapshot, watchlist_snapshot, daily_report_file):
    """渲染首页概览及最新日报入口。"""
    st.header("首页")
    report_date = quant_snapshot.get("快照日期", "未提供")
    watch_stocks = watchlist_snapshot.get("stocks", []) if isinstance(watchlist_snapshot, dict) else []
    matched = watchlist_snapshot.get("matched", "未提供") if isinstance(watchlist_snapshot, dict) else "未提供"
    missing = watchlist_snapshot.get("missing", "未提供") if isinstance(watchlist_snapshot, dict) else "未提供"
    columns = st.columns(4)
    columns[0].metric("最新量化快照日期", report_date)
    columns[1].metric("关注股数量", len(watch_stocks))
    columns[2].metric("匹配数量", matched)
    columns[3].metric("缺失数量", missing)

    st.subheader("最近日报")
    if not daily_report_file:
        st.info("未找到每日关注股票日报。请先完成日报生成流程。")
        return
    mtime = datetime.fromtimestamp(daily_report_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"生成时间：{mtime}")
    st.code(str(daily_report_file), language=None)
    try:
        report_content = load_text_cached(str(daily_report_file), get_file_mtime_ns(daily_report_file))
    except ValueError as error:
        st.warning(str(error))
        return
    with st.expander("查看当天日报内容", expanded=False):
        st.markdown(report_content)
    st.download_button(
        "下载当天日报 Markdown",
        data=report_content,
        file_name=daily_report_file.name,
        mime="text/markdown",
    )


def render_stock_details(stock_record, daily_signal=None, signal_override=None):
    """展示单股票已有事实、日间变化、规则结论和按需 AI 解释。"""
    fact_snapshot = build_fact_snapshot(stock_record)
    signal_stock = signal_override or get_signal_stock(daily_signal, stock_record)
    st.subheader(f"{fact_snapshot['股票名称']}（{fact_snapshot['股票代码']}）")
    st.caption(f"别名：{fact_snapshot['别名'] or '未设置'} ｜ 标签：{'、'.join(fact_snapshot['标签']) or '未设置'} ｜ 优先级：{fact_snapshot['优先级'] if fact_snapshot['优先级'] is not None else '未设置'}")

    metrics = st.columns(5)
    metrics[0].metric("Score", fact_snapshot["综合评分"])
    metrics[1].metric("RSI", fact_snapshot["RSI"])
    metrics[2].metric("MA5", fact_snapshot["MA5"])
    metrics[3].metric("MA20", fact_snapshot["MA20"])
    metrics[4].metric("MACD", fact_snapshot["MACD"])
    st.write(f"趋势：{fact_snapshot['趋势']} ｜ 建议：{fact_snapshot['建议']} ｜ 风险标签：{fact_snapshot['风险标签']}")

    st.markdown("#### 今日变化")
    if signal_stock:
        changes = signal_stock.get("今日变化", {})
        st.write(f"数据状态：{signal_stock.get('数据状态', '数据不足')}")
        change_columns = st.columns(4)
        change_columns[0].metric("Score 变化", changes.get("Score变化", "数据不足"))
        change_columns[1].metric("RSI 变化", changes.get("RSI变化", "数据不足"))
        change_columns[2].write(f"MA5/MA20：{changes.get('MA5/MA20关系变化', '数据不足')}")
        change_columns[3].write(f"MACD：{changes.get('MACD状态变化', '数据不足')}")
        st.write(f"信号分类：{signal_stock.get('信号分类', '观察')}")
        st.markdown("#### 观察重点")
        for condition in signal_stock.get("观察重点", []):
            st.markdown(f"- {condition}")
    else:
        st.info("未找到与当前量化快照同日期的 daily_signal，今日变化数据不足。")

    rule_summary = build_rule_summary(fact_snapshot)
    st.markdown("#### 规则分析")
    st.write(rule_summary)

    stock_evidence, market_context = get_stock_evidence(stock_record)
    st.markdown("#### 证据解读")
    st.write(f"市场数据截至：{market_context.get('数据截至日期', '数据不足')}")
    st.markdown("偏强证据：")
    for item in stock_evidence.get("偏强证据", ["数据不足"]):
        st.markdown(f"- {item}")
    st.markdown("谨慎证据：")
    for item in stock_evidence.get("谨慎证据", ["数据不足"]):
        st.markdown(f"- {item}")

    ai_key = f"ai_markdown_{fact_snapshot['股票代码'] or fact_snapshot['股票名称']}"
    if st.button("生成/刷新 AI 分析", key=f"generate_{ai_key}"):
        ai_summary = build_ai_summary({**stock_evidence, "市场环境": market_context})
        st.session_state[ai_key] = create_evidence_markdown_content(stock_evidence, market_context, ai_summary)
    if ai_key in st.session_state:
        st.markdown("#### AI 分析报告")
        st.markdown(st.session_state[ai_key])


def render_on_demand_analysis(analysis):
    """展示隔离按需分析快照，并仅在明确点击时写入关注列表。"""
    stock_record = analysis.get("stock_record")
    if not isinstance(stock_record, dict):
        st.warning("按需分析快照格式错误，无法展示。")
        return
    st.caption(
        f"数据日期：{analysis.get('数据日期', '未提供')} ｜ "
        f"{analysis.get('日线数据说明', '仅供日线分析。')}"
    )
    st.caption(
        f"行情来源：{analysis.get('行情来源', '未记录')} ｜ "
        f"名称目录来源：{analysis.get('名称目录来源', '未记录')}"
    )
    render_stock_details(stock_record, signal_override=analysis.get("daily_signal"))
    if st.button("加入关注列表", key=f"watch_{stock_record['code']}"):
        result = add_stock_to_watchlist(stock_record)
        if result["status"] == "success":
            st.success(result["message"])
        elif result["status"] == "exists":
            st.info(result["message"])
        else:
            st.error(result["message"])


def render_on_demand_candidate(stock):
    """展示单一按需候选及其隔离缓存，并等待用户明确触发下载。"""
    result_message_key = f"on_demand_result_{stock['code']}"
    result_message = st.session_state.pop(result_message_key, None)
    if result_message:
        st.success(result_message)

    cached_analysis = load_on_demand_snapshot(stock["code"])
    if cached_analysis:
        st.info("已找到该股票的本地按需分析缓存；可直接查看或点击“刷新数据”。")
        render_on_demand_analysis(cached_analysis)
        refresh_label = "刷新数据"
    else:
        st.info("点击“下载并分析”后才会请求这一只股票的腾讯历史日线。")
        refresh_label = "下载并分析"

    if st.button(refresh_label, key=f"download_{stock['code']}"):
        with st.spinner("正在通过腾讯接口下载单股历史日线并按现有规则分析…"):
            result = analyze_on_demand_stock(stock, refresh=cached_analysis is not None)
        if result["status"] in {"success", "cached"}:
            # 当前轮已因缓存展示过一次详情；直接再次展示会创建重复的 AI
            # 按钮 key。重新运行后从新快照读取，并且每只股票只渲染一次详情。
            st.session_state[result_message_key] = result["message"]
            st.rerun()
        else:
            st.error(result["message"])


def render_on_demand_query(query):
    """六码直查绕过名称目录；名称检索仅使用已有本地目录。"""
    direct_stock = resolve_code_query(query)
    if direct_stock:
        st.caption("六码代码直查：不依赖名称目录；点击下载后将通过腾讯接口识别股票并获取日线。")
        render_on_demand_candidate(direct_stock)
        return

    try:
        catalog = load_catalog_cached(str(CATALOG_FILE), get_file_mtime_ns(CATALOG_FILE))
    except ValueError as error:
        st.warning(f"名称目录不可用，可输入六位股票代码继续分析。详情：{error}")
        catalog = []

    st.caption(f"本地 A 股代码目录：{len(catalog)} 条。目录更新只获取名称和代码，不下载日线。")
    if st.button("更新本地代码目录（可选）", key="refresh_catalog"):
        with st.spinner("正在更新本地 A 股代码目录…"):
            result = refresh_catalog()
        if result["status"] == "success":
            st.success(f"{result['message']} 当前共 {result['count']} 条。")
            catalog = load_catalog(CATALOG_FILE)
        else:
            st.warning(f"{result['message']} 名称目录不可用，可输入六位股票代码继续分析。")

    candidates = resolve_catalog_query(query, catalog)
    if not candidates:
        st.warning("名称目录不可用或未找到该名称，可输入六位股票代码继续分析。")
        return
    selected_index = 0
    if len(candidates) > 1:
        selected_index = st.radio(
            "找到多个目录候选股票，请点击选择：",
            range(len(candidates)),
            format_func=lambda index: f"{candidates[index]['name']}（{candidates[index]['code']}）",
            key="on_demand_candidate",
        )
    stock = {**candidates[selected_index], "名称目录来源": "本地 A 股代码名称目录缓存"}
    render_on_demand_candidate(stock)


def render_stock_query(stock_records, daily_signal):
    """先复用已有快照，再为其他 A 股提供隔离的按需查询入口。"""
    st.header("单股票查询")
    query = st.text_input("输入股票代码或名称", placeholder="例如：600839、四川长虹、长虹")
    if not query.strip():
        st.info("请输入股票代码或名称进行查询。")
        return

    stock_record, error_message = find_stock(query, stock_records)
    if error_message == "找到多个股票，请输入完整名称或代码。":
        candidates = find_fuzzy_matches(query, stock_records)
        selected_index = st.radio(
            "找到多个候选股票，请点击选择：",
            range(len(candidates)),
            format_func=lambda index: f"{candidates[index]['name']}（{candidates[index]['code']}）",
        )
        stock_record = candidates[selected_index]
    elif error_message:
        if error_message != "未找到该股票。":
            st.warning(error_message)
            return
        render_on_demand_query(query)
        return

    render_stock_details(stock_record, daily_signal)


def render_watchlist(watchlist_data):
    """渲染只读关注列表，不提供页面内编辑入口。"""
    st.header("关注列表（只读）")
    rows = get_watchlist_rows(watchlist_data)
    if not rows:
        st.info("watchlist.json 中没有可展示的关注股票。")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("首版仅展示 watchlist.json，不在页面直接修改关注配置。")


def main():
    """启动本机 Streamlit 页面。"""
    if st is None:
        raise RuntimeError("未安装 streamlit，请先执行 .venv/bin/pip install -r requirements.txt。")

    st.set_page_config(page_title="AStockAI 本地查询", page_icon="📈", layout="wide")
    st.title("AStockAI 本地 Web 查询窗口")
    st.caption("已有快照只读展示；非快照股票仅在明确点击后下载单股日线。不会自动发送邮件或运行回测。 ")

    quant_file = find_latest_snapshot_file(OUTPUT_DIRECTORY, "quant_snapshot")
    if quant_file is None:
        st.error("未找到 quant_snapshot。请先运行量化研究和日报流程。")
        return
    watch_snapshot_file = find_latest_snapshot_file(OUTPUT_DIRECTORY, "watchlist_snapshot")
    if watch_snapshot_file is None:
        st.warning("未找到 watchlist_snapshot；单股票查询将仅使用 quant_snapshot。")

    try:
        quant_snapshot = load_json_cached(str(quant_file), get_file_mtime_ns(quant_file), "最新 quant_snapshot")
        watchlist_snapshot = (
            load_json_cached(str(watch_snapshot_file), get_file_mtime_ns(watch_snapshot_file), "最新 watchlist_snapshot")
            if watch_snapshot_file
            else None
        )
        stock_records = load_stock_records_cached(
            str(OUTPUT_DIRECTORY),
            get_file_mtime_ns(quant_file),
            get_file_mtime_ns(watch_snapshot_file),
        )
        watchlist_data = load_json_cached(str(WATCHLIST_FILE), get_file_mtime_ns(WATCHLIST_FILE), "watchlist.json")
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    report_date = quant_snapshot.get("快照日期", "未知日期")
    signal_file = OUTPUT_DIRECTORY / f"daily_signal_{report_date}.json"
    try:
        daily_signal = (
            load_json_cached(str(signal_file), get_file_mtime_ns(signal_file), "daily_signal")
            if signal_file.is_file()
            else None
        )
    except ValueError as error:
        st.warning(f"daily_signal 不可用：{error}")
        daily_signal = None

    home_tab, query_tab, watchlist_tab = st.tabs(["首页", "单股票查询", "关注列表"])
    with home_tab:
        render_home(quant_snapshot, watchlist_snapshot, get_report_file(report_date))
    with query_tab:
        render_stock_query(stock_records, daily_signal)
    with watchlist_tab:
        render_watchlist(watchlist_data)


if __name__ == "__main__":
    main()
