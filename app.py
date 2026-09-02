"""AStockAI v4.4：本机只读量化查询窗口。"""

import json
from datetime import datetime
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:  # 允许在未安装 Streamlit 的测试环境导入数据辅助函数。
    st = None

from astock_core.analysis.stock_analysis import (
    build_ai_summary,
    build_fact_snapshot,
    build_rule_summary,
    create_markdown_content,
    create_evidence_markdown_content,
    create_stock_records,
    enrich_stock_evidence_with_research,
    find_fuzzy_matches,
    find_latest_snapshot_file,
    find_stock,
    load_quant_snapshot,
    load_watchlist_snapshot,
)
from astock_core.analysis.analysis_evidence import build_report_evidence, build_stock_evidence
from astock_core.analysis.on_demand_analysis import (
    CATALOG_FILE,
    add_stock_to_watchlist,
    analyze_on_demand_stock,
    load_catalog,
    load_on_demand_snapshot,
    refresh_catalog,
    resolve_code_query,
    resolve_catalog_query,
)
from astock_core.analysis.fundamental_data import (
    build_industry_peer_comparison,
    build_valuation_observation,
    collect_fundamental_snapshot,
    load_fundamental_snapshot,
    summarize_fundamental_evidence,
)
from astock_core.analysis.expert_research import build_expert_research_memo
from astock_core.research.research_dashboard import (
    build_research_dashboard,
    build_research_workbench_summary,
    build_user_system_status,
    research_dashboard_source_mtime,
)
from astock_core.runtime.background_tasks import (
    get_active_task,
    list_task_statuses,
    read_task_log_tail,
    start_background_task,
)
from astock_core.portfolio.portfolio_management import (
    build_investment_review,
    build_portfolio_rows,
    load_portfolio,
    remove_holding,
    save_portfolio,
    summarize_portfolio,
    upsert_cash,
    upsert_holding,
)
from astock_core.analysis.five_day_risk_range import build_five_day_risk_range
from astock_core.strategies.afternoon_momentum import (
    STRATEGY_ID,
    load_latest_strategy_run,
    run_afternoon_momentum_screen,
    save_strategy_run,
    strategy_catalog,
)
from astock_core.simulator.paper_portfolio import (
    create_snapshot_buy,
    load_simulator,
    save_simulator,
    summarize_simulator,
    upsert_simulator_cash,
)


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output"
WATCHLIST_FILE = PROJECT_DIRECTORY / "watchlist.json"
PORTFOLIO_FILE = PROJECT_DIRECTORY / "data" / "portfolio.json"
SIMULATOR_FILE = PROJECT_DIRECTORY / "data" / "simulator.json"
RAW_INTERVAL_DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "raw_interval"
INTERVAL_RESEARCH_DIRECTORY = OUTPUT_DIRECTORY / "research"
FUNDAMENTAL_METRICS = (
    "营业总收入",
    "归母净利润",
    "营业总收入同比增长",
    "归母净利润同比增长",
    "扣非净利润同比增长",
    "净资产收益率(加权)",
    "销售毛利率",
    "资产负债率",
    "每股经营现金流",
    "每股收益(基本)",
    "每股净资产",
)


def build_fundamental_display_data(stock_evidence):
    """把已保存的基本面研究事实整理为页面展示数据，不推导投资结论。"""
    fundamental = stock_evidence.get("基本面研究证据", {}) if isinstance(stock_evidence, dict) else {}
    valuation = stock_evidence.get("估值观察", {}) if isinstance(stock_evidence, dict) else {}
    peer = stock_evidence.get("行业同业比较", {}) if isinstance(stock_evidence, dict) else {}
    result = {
        "数据状态": fundamental.get("数据状态", "数据不足：未下载基本面快照。"),
        "报告期": fundamental.get("报告期", "未提供"),
        "公告日期": fundamental.get("公告日期", "未提供"),
        "来源": fundamental.get("来源", "未提供"),
        "官方核验页": fundamental.get("官方核验页"),
        "财务指标": [],
        "公司与行业事实": [],
        "估值观察": valuation,
        "同业比较": peer,
    }
    if result["数据状态"] != "可用":
        return result

    metrics = fundamental.get("指标", {})
    result["财务指标"] = [
        {"指标": label, "数值": metrics[label], "口径": "来源原始口径"}
        for label in FUNDAMENTAL_METRICS
        if metrics.get(label) is not None
    ]
    result["公司与行业事实"] = [
        fact for fact in fundamental.get("事实", [])
        if str(fact).startswith(("所属行业：", "证监会行业：", "主营业务：", "上市市场：", "实际控制人："))
    ]
    return result


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
def load_research_dashboard_cached(project_directory, source_mtime_ns):
    """按研究产物的最新修改时间缓存总览；只读取本地 JSON。"""
    return build_research_dashboard(project_directory)


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
    stock_evidence = build_stock_evidence(
        stock_record, *contexts,
        {"available": True, "stocks": []}, evidence["市场环境"],
    )
    return enrich_stock_evidence_with_research(stock_evidence, PROJECT_DIRECTORY), evidence["市场环境"]


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


def render_fundamental_research(stock_evidence):
    """渲染本地基本面快照、机械估值观察和有限同业比较。"""
    display = build_fundamental_display_data(stock_evidence)
    st.markdown("#### 基本面事实快照（本地）")
    if display["数据状态"] != "可用":
        st.info(display["数据状态"])
        st.caption("可点击页面顶部的“下载/刷新基本面快照”后再查看；本页不会把缺失数据补成结论。")
        return

    st.caption(f"报告期：{display['报告期']} ｜ 公告日期：{display['公告日期']}")
    if display["财务指标"]:
        st.dataframe(display["财务指标"], use_container_width=True, hide_index=True)
    else:
        st.info("最新报告未提供可展示的主要财务指标。")

    if display["公司与行业事实"]:
        with st.expander("公司与行业原始描述", expanded=False):
            for fact in display["公司与行业事实"]:
                st.markdown(f"- {fact}")

    valuation = display["估值观察"]
    st.markdown("**估值观察（机械计算）**")
    if valuation.get("数据状态") == "可用":
        columns = st.columns(3)
        columns[0].metric("本地最新收盘", valuation.get("最新收盘", "数据不足"))
        columns[1].metric("PB", valuation.get("市净率(PB)") if valuation.get("市净率(PB)") is not None else "数据不足")
        static_pe = valuation.get("静态市盈率(PE)")
        columns[2].metric("静态 PE", static_pe if static_pe is not None else "不适用（非年报）")
        st.caption(valuation.get("说明", "PB/PE 仅是已披露每股指标与本地收盘价的机械比值。"))
    else:
        st.caption(valuation.get("数据状态", "数据不足：无法计算估值观察。"))

    peer = display["同业比较"]
    st.markdown("**同业比较（本地已下载快照）**")
    if peer.get("数据状态") == "可用":
        peer_rows = []
        for label, item in peer.get("指标比较", {}).items():
            if item.get("数据状态") == "可用":
                peer_rows.append({
                    "指标": label,
                    "本公司": item["本公司"],
                    "同业中位数": item["同业中位数"],
                    "排名": f"{item['同业排名']}/{item['有效可比公司数']}",
                    "比较方向": item["方向"],
                })
        st.caption(f"行业：{peer.get('所属行业', '未提供')} ｜ 同报告期本地可比：{peer.get('可比公司数量', 0)} 家")
        if peer_rows:
            st.dataframe(peer_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("同业快照中没有足够的可比指标。")
        st.caption(peer.get("说明", "仅使用同一行业、同一报告期的本地快照。"))
    else:
        st.caption(peer.get("数据状态", "数据不足：无法进行同业比较。"))

    source_text = f"数据来源：{display['来源']}。"
    if display["官方核验页"]:
        source_text += f"请以[巨潮资讯官方定期报告]({display['官方核验页']})复核口径。"
    st.caption(source_text)


def get_report_file(report_date):
    """优先返回与最新量化日期对应的日报，缺失时返回已有最新日报。"""
    dated_file = OUTPUT_DIRECTORY / f"每日关注股票日报_{report_date}.md"
    return dated_file if dated_file.is_file() else find_latest_file(OUTPUT_DIRECTORY, "每日关注股票日报_*.md")


def render_home(quant_snapshot, watchlist_snapshot, daily_report_file, daily_signal):
    """围绕本地持仓渲染投资总览，运行状态只作为次级资料覆盖信息。"""
    st.header("投资总览")
    report_date = quant_snapshot.get("快照日期", "未提供")
    try:
        portfolio = load_portfolio(PORTFOLIO_FILE)
    except ValueError as error:
        st.error(str(error))
        return
    rows = build_portfolio_rows(
        portfolio,
        _local_quote_map(quant_snapshot),
        daily_signal.get("stocks", []) if isinstance(daily_signal, dict) else [],
    )
    summary = summarize_portfolio(rows, portfolio)
    if rows:
        assets = summary["已报价持仓市值"] + summary["现金余额"]
        top_weight = max(
            (row["当前市值"] or 0 for row in rows), default=0
        ) / summary["已报价持仓市值"] * 100 if summary["已报价持仓市值"] else None
        columns = st.columns(5)
        columns[0].metric("本地资产合计", _money(assets))
        columns[1].metric("已报价持仓市值", _money(summary["已报价持仓市值"]))
        columns[2].metric("现金余额", _money(summary["现金余额"]))
        columns[3].metric("已报价浮盈亏", _money(summary["已报价浮盈亏"]))
        columns[4].metric("最大单一持仓占比", f"{top_weight:.2f}%" if top_weight is not None else "数据不足")
        if summary["缺少本地报价数"]:
            st.warning(f"{summary['缺少本地报价数']} 只实际持仓未匹配本地快照；请先更新数据，再对市值或研究状态作判断。")

        st.subheader("今天的持仓核对清单")
        st.caption(f"研究快照日期：{report_date}。这是已有事实的核对顺序，不是买卖建议。")
        review_rows = build_investment_review(rows, daily_signal.get("stocks", []) if isinstance(daily_signal, dict) else [])
        st.dataframe(
            review_rows,
            use_container_width=True,
            hide_index=True,
            column_config={"持仓占比": st.column_config.NumberColumn(format="%.2f%%")},
        )
    else:
        st.info("尚未录入本地持仓；投资总览会在你确认保存首条持仓后显示资金暴露、数据缺口和今日核对清单。")

    with st.expander("系统资料覆盖", expanded=False):
        st.caption("以下是报告资料的覆盖情况，不等同于你的持仓结论。")
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


def _format_percent(value):
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) and not isinstance(value, bool) else "数据不足"


def _format_metric(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "数据不足"


def render_background_task_status():
    """展示最近后台任务的持久化状态和有限日志尾部。"""
    active_task = get_active_task(PROJECT_DIRECTORY)
    recent_tasks = list_task_statuses(PROJECT_DIRECTORY)
    if active_task:
        st.info(
            f"后台任务运行中：{active_task.get('任务类型')}（{active_task.get('任务编号')}）。"
            "状态会自动刷新。"
        )
    if not recent_tasks:
        st.caption("尚无后台任务记录。")
        return
    latest = recent_tasks[0]
    latest_state = (latest.get("任务编号"), latest.get("状态"))
    previous_state = st.session_state.get("last_background_task_state")
    st.session_state["last_background_task_state"] = latest_state
    if (
        previous_state
        and previous_state[0] == latest_state[0]
        and previous_state[1] in {"queued", "running"}
        and latest_state[1] not in {"queued", "running"}
    ):
        # 任务结束后完整重跑页面，使总览缓存按新产物的修改时间失效。
        st.rerun()
    rows = [
        {
            "任务": task.get("任务类型", "未知"),
            "状态": task.get("状态", "未知"),
            "开始时间": task.get("开始时间", "未提供"),
            "结束时间": task.get("结束时间", "运行中"),
            "说明": task.get("说明", ""),
        }
        for task in recent_tasks
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander(f"查看最新任务详情：{latest.get('任务编号', '未知')}", expanded=False):
        if latest.get("步骤"):
            st.dataframe(latest["步骤"], use_container_width=True, hide_index=True)
        if latest.get("边界"):
            st.caption(latest["边界"])
        log_tail = read_task_log_tail(PROJECT_DIRECTORY, latest)
        if log_tail:
            st.code(log_tail, language="text")


def render_background_task_controls():
    """提供显式后台操作；任一任务执行时禁止启动另一个写入任务。"""
    active_task = get_active_task(PROJECT_DIRECTORY)
    st.markdown("#### 后台更新与研究任务")
    st.caption("数据更新会访问腾讯行情源；研究重建只使用本地数据。两类任务均在独立进程中运行。")
    columns = st.columns(2)
    if columns[0].button("后台更新本地行情", key="start_data_update", disabled=active_task is not None):
        result = start_background_task("data_update", PROJECT_DIRECTORY)
        if result["status"] == "started":
            st.success(result["message"])
            st.rerun()
        else:
            st.error(result["message"])
    if columns[1].button("后台重建研究产物", key="start_research_refresh", disabled=active_task is not None):
        result = start_background_task("research_refresh", PROJECT_DIRECTORY)
        if result["status"] == "started":
            st.success(result["message"])
            st.rerun()
        else:
            st.error(result["message"])
    if active_task:
        st.caption("为避免行情写入与模型训练读取同一批文件，当前任务结束前不能启动另一任务。")


def render_research_dashboard(dashboard):
    """渲染用户优先的系统状态；研究工程细节默认收起。"""
    st.header("系统状态")
    st.caption("这里只说明系统是否已经准备好供你查看，不提供买卖建议。")
    user_status = build_user_system_status(dashboard)
    notice = getattr(st, user_status["提示级别"], st.info)
    notice(f"**{user_status['标题']}**\n\n{user_status['说明']}\n\n{user_status['建议动作']}")
    columns = st.columns(2)
    columns[0].metric("日常数据", user_status["数据状态"])
    columns[1].metric("模型预测", user_status["模型状态"])
    st.caption(user_status["模型说明"])

    with st.expander("手动刷新数据（通常不需要）", expanded=False):
        st.caption("只有数据不足或你明确希望获取新日线时才使用；不会发送邮件或执行交易。")
        render_background_task_controls()

        @st.fragment(run_every=3)
        def task_status_fragment():
            render_background_task_status()

        task_status_fragment()

    health = dashboard.get("数据健康", {})
    with st.expander("高级信息：数据与模型检查", expanded=False):
        st.caption("以下内容用于排查研究工程，不是日常持仓决策所必需的信息。")
        if health.get("状态") == "可用":
            columns = st.columns(4)
            columns[0].metric("可训练股票", health.get("可训练股票数", "数据不足"))
            columns[1].metric("最新日线日期", health.get("最新日线日期", "数据不足"))
            columns[2].metric("数据问题文件", health.get("数据问题文件数", "数据不足"))
            columns[3].metric("研究池外文件", health.get("研究池外文件数", "数据不足"))
            st.caption(health.get("说明", ""))
            st.dataframe(health.get("文件审计", []), use_container_width=True, hide_index=True)
            if health.get("特征构建跳过文件"):
                st.write("特征构建跳过文件：")
                st.dataframe(health["特征构建跳过文件"], use_container_width=True, hide_index=True)
        else:
            st.info(health.get("说明", "尚无数据审计报告。"))

        model = dashboard.get("模型验证", {})
        st.markdown("#### v5.1 样本外验证")
        if model.get("状态") == "数据不足":
            st.info(model.get("说明", "尚无样本外验证报告。"))
        else:
            columns = st.columns(4)
            columns[0].metric("训练股票", model.get("训练股票数", "数据不足"))
            columns[1].metric("训练截止日期", model.get("训练截止日期", "数据不足"))
            columns[2].metric("完成/校准窗口", f"{model.get('完成窗口数', 0)}/{model.get('完成校准窗口数', 0)}")
            metrics = model.get("样本外指标", {})
            baseline = model.get("朴素概率基线", {})
            columns[3].metric("样本外 ROC-AUC", _format_metric(metrics.get("roc_auc")))
            metric_rows = [
                {"指标": "Brier Score", "模型": _format_metric(metrics.get("brier_score")), "朴素基线": _format_metric(baseline.get("brier_score"))},
                {"指标": "Log Loss", "模型": _format_metric(metrics.get("log_loss")), "朴素基线": _format_metric(baseline.get("log_loss"))},
                {"指标": "ROC-AUC", "模型": _format_metric(metrics.get("roc_auc")), "朴素基线": _format_metric(baseline.get("roc_auc"))},
                {"指标": "准确率", "模型": _format_metric(metrics.get("accuracy")), "朴素基线": _format_metric(baseline.get("accuracy"))},
            ]
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)
            st.markdown("**验证风险提示**")
            for risk in model.get("风险提示", []):
                st.markdown(f"- {risk}")

        portfolio = dashboard.get("组合回测", {})
        st.markdown("#### 滚动样本外组合实验")
        if portfolio.get("状态") != "可用":
            st.info(portfolio.get("说明", "尚无严格滚动样本外组合回测报告。"))
        else:
            statistics = portfolio.get("统计", {})
            columns = st.columns(4)
            columns[0].metric("策略累计收益", _format_percent(statistics.get("累计收益率")))
            columns[1].metric("相对基准超额", _format_percent(statistics.get("超额累计收益率")))
            columns[2].metric("最大回撤", _format_percent(statistics.get("最大回撤")))
            columns[3].metric("交易笔数", statistics.get("交易笔数", "数据不足"))
            st.caption(f"策略：{portfolio.get('策略', '未提供')} ｜ 基准：{portfolio.get('市场基准', '未提供')}。{portfolio.get('说明', '')}")
            st.json({
                "参数": portfolio.get("参数", {}),
                "信号覆盖诊断": portfolio.get("信号覆盖诊断", {}),
                "跳过文件": portfolio.get("跳过文件", []),
            })

        st.caption("研究来源：" + " ｜ ".join(
            path for path in (
                health.get("报告文件"), model.get("报告文件"), portfolio.get("报告文件")
            ) if path
        ))


def render_stock_details(stock_record, daily_signal=None, signal_override=None, research_override=None):
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

    risk_range = build_five_day_risk_range(
        {"code": fact_snapshot["股票代码"], "name": fact_snapshot["股票名称"]},
        RAW_INTERVAL_DATA_DIRECTORY,
        INTERVAL_RESEARCH_DIRECTORY,
    )
    if risk_range["状态"] == "可用":
        st.markdown("#### 未来 5 日风险范围")
        range_columns = st.columns(3)
        range_columns[0].metric("常见收盘范围", f"¥{risk_range['下限价格']:.3f} – ¥{risk_range['上限价格']:.3f}")
        range_columns[1].metric("最近本地收盘", f"¥{risk_range['本地收盘']:.3f}")
        range_columns[2].metric("历史覆盖", f"约 {risk_range['历史覆盖率']:.1f}%")
        st.caption(
            f"数据截至 {risk_range['数据日期']}。{risk_range['说明']} "
            f"验证样本 {risk_range['验证样本数']:,} 个；{risk_range['边界']}"
        )
    else:
        st.caption(f"未来 5 日风险范围暂不可用：{risk_range['说明']}")

    if st.button("下载/刷新基本面快照", key=f"fundamentals_{fact_snapshot['股票代码']}"):
        with st.spinner("正在下载最近报告期基本面事实…"):
            result = collect_fundamental_snapshot(fact_snapshot["股票代码"])
        if result["status"] == "success":
            st.success(result["message"])
            st.rerun()
        else:
            st.error(result["message"])

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
    if isinstance(research_override, dict):
        stock_evidence = {**stock_evidence, **research_override}
    st.markdown("#### 证据解读")
    st.write(f"市场数据截至：{market_context.get('数据截至日期', '数据不足')}")
    st.markdown("偏强证据：")
    for item in stock_evidence.get("偏强证据", ["数据不足"]):
        st.markdown(f"- {item}")
    st.markdown("谨慎证据：")
    for item in stock_evidence.get("谨慎证据", ["数据不足"]):
        st.markdown(f"- {item}")

    render_fundamental_research(stock_evidence)

    memo = stock_evidence.get("专家研究备忘录", {})
    st.markdown("#### 专家研究框架（证据化）")
    st.write(memo.get("核心研究论点", "数据不足"))
    for heading in ("支持证据", "相反证据与风险", "待验证或证伪"):
        st.markdown(f"**{heading}**")
        for item in memo.get(heading, ["数据不足"]):
            st.markdown(f"- {item}")
    st.caption(memo.get("基本面与行业证据", "数据不足"))

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
    fundamental_snapshot = load_fundamental_snapshot(stock_record.get("code", ""))
    fundamental_evidence = summarize_fundamental_evidence(fundamental_snapshot)
    price_evidence = analysis.get("价格研究证据", {})
    fundamental_evidence["价格日期"] = price_evidence.get("数据截至日期", "数据不足")
    valuation_evidence = build_valuation_observation(
        fundamental_evidence, price_evidence.get("最新收盘")
    )
    peer_comparison = build_industry_peer_comparison(fundamental_snapshot)
    memo = build_expert_research_memo(
        {"当前量化证据": analysis.get("daily_signal", {}).get("当前指标", {})},
        price_evidence,
        fundamental_evidence,
        valuation_evidence,
        peer_comparison,
    )
    render_stock_details(
        stock_record,
        signal_override=analysis.get("daily_signal"),
        research_override={
            "价格研究证据": price_evidence,
            "基本面研究证据": fundamental_evidence,
            "估值观察": valuation_evidence,
            "行业同业比较": peer_comparison,
            "专家研究备忘录": memo,
        },
    )
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


def _local_quote_map(quant_snapshot):
    """从本地量化快照提取最近收盘；页面不得把日线收盘标成实时行情。"""
    quote_date = str(quant_snapshot.get("快照日期", "数据不足"))
    quotes = {}
    for stock in quant_snapshot.get("股票排行榜", []) if isinstance(quant_snapshot, dict) else []:
        if not isinstance(stock, dict):
            continue
        code = str(stock.get("股票代码", "")).strip().zfill(6)
        close = stock.get("收盘价")
        if code.isdigit() and len(code) == 6:
            quotes[code] = {"close": close, "date": quote_date, "advice": stock.get("建议", "数据不足")}
    return quotes


def _money(value):
    return f"¥{value:,.2f}" if isinstance(value, (int, float)) else "数据不足"


def render_portfolio(quant_snapshot, daily_signal):
    """渲染仅本地保存的持仓台账，研究数据与实际账户数据保持隔离。"""
    st.header("持仓管理（本地）")
    st.caption("账户、成本与数量只保存于本机 data/portfolio.json（已被 Git 忽略）。行情为本地量化快照中的最近收盘，不是实时盘中行情。")
    try:
        portfolio = load_portfolio(PORTFOLIO_FILE)
    except ValueError as error:
        st.error(str(error))
        return

    quotes = _local_quote_map(quant_snapshot)
    signal_stocks = daily_signal.get("stocks", []) if isinstance(daily_signal, dict) else []
    rows = build_portfolio_rows(portfolio, quotes, signal_stocks)
    summary = summarize_portfolio(rows, portfolio)
    metrics = st.columns(5)
    metrics[0].metric("持仓标的", summary["持仓数量"])
    metrics[1].metric("持仓成本", _money(summary["持仓成本"]))
    metrics[2].metric("已报价市值", _money(summary["已报价持仓市值"]))
    metrics[3].metric("已报价浮盈亏", _money(summary["已报价浮盈亏"]))
    metrics[4].metric("现金余额", _money(summary["现金余额"]))
    if summary["缺少本地报价数"]:
        st.warning(f"{summary['缺少本地报价数']} 只持仓未匹配本地量化快照，未估算其市值或浮盈亏。")

    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "平均成本": st.column_config.NumberColumn(format="¥%.3f"),
                "本地最近收盘": st.column_config.NumberColumn(format="¥%.3f"),
                "持仓成本": st.column_config.NumberColumn(format="¥%.2f"),
                "当前市值": st.column_config.NumberColumn(format="¥%.2f"),
                "浮盈亏": st.column_config.NumberColumn(format="¥%.2f"),
                "浮盈亏率": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    else:
        st.info("尚未录入持仓。下面可新增第一条本地持仓记录。")

    stock_options = [stock for stock in quant_snapshot.get("股票排行榜", []) if isinstance(stock, dict)]
    with st.expander("新增或更新持仓", expanded=not rows):
        if not stock_options:
            st.info("缺少本地量化快照，暂不能从股票池选择标的。")
        else:
            labels = [f"{stock.get('股票名称', '未知')}（{str(stock.get('股票代码', '')).zfill(6)}）" for stock in stock_options]
            with st.form("portfolio_holding_form", clear_on_submit=False):
                account = st.text_input("账户名称", placeholder="例如：普通账户")
                choice = st.selectbox("股票", range(len(stock_options)), format_func=lambda index: labels[index])
                quantity = st.number_input("持仓数量", min_value=1, step=100)
                cost_price = st.number_input("平均成本", min_value=0.0, step=0.001, format="%.3f")
                category = st.text_input("类别（可选）", placeholder="例如：长期观察")
                if st.form_submit_button("生成待确认草稿"):
                    selected = stock_options[choice]
                    try:
                        preview = upsert_holding(
                            {"version": "1.0", "holdings": [], "cash": []},
                            {
                                "account": account,
                                "code": selected.get("股票代码"),
                                "name": selected.get("股票名称"),
                                "quantity": quantity,
                                "cost_price": cost_price,
                                "category": category,
                            },
                        )["holdings"][0]
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.session_state["portfolio_holding_draft"] = preview

            draft = st.session_state.get("portfolio_holding_draft")
            if isinstance(draft, dict):
                st.markdown("#### 待确认持仓")
                st.dataframe([draft], use_container_width=True, hide_index=True)
                confirmed = st.checkbox("我已核对账户、股票代码、数量和平均成本，确认写入本地账本。", key="portfolio_holding_confirm")
                confirm_column, cancel_column = st.columns(2)
                if confirm_column.button("确认写入本地持仓", disabled=not confirmed, type="primary"):
                    try:
                        save_portfolio(PORTFOLIO_FILE, upsert_holding(portfolio, draft))
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.session_state.pop("portfolio_holding_draft", None)
                        st.session_state.pop("portfolio_holding_confirm", None)
                        st.success("本地持仓已确认保存。")
                        st.rerun()
                if cancel_column.button("放弃此草稿"):
                    st.session_state.pop("portfolio_holding_draft", None)
                    st.session_state.pop("portfolio_holding_confirm", None)
                    st.rerun()

    with st.expander("更新现金余额", expanded=False):
        with st.form("portfolio_cash_form", clear_on_submit=True):
            cash_account = st.text_input("账户名称", key="cash_account")
            cash_amount = st.number_input("现金余额", min_value=0.0, step=100.0)
            if st.form_submit_button("保存本地现金"):
                try:
                    save_portfolio(PORTFOLIO_FILE, upsert_cash(portfolio, cash_account, cash_amount))
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("本地现金余额已保存。")
                    st.rerun()

    if portfolio.get("holdings"):
        with st.expander("删除持仓", expanded=False):
            delete_options = [f"{item['account']}｜{item['name']}（{item['code']}）" for item in portfolio["holdings"]]
            selected_delete = st.selectbox("选择要删除的持仓", range(len(portfolio["holdings"]),), format_func=lambda index: delete_options[index])
            delete_confirmed = st.checkbox("我确认删除这条本地持仓记录。", key="portfolio_delete_confirm")
            if st.button("删除所选持仓", type="secondary", disabled=not delete_confirmed):
                selected = portfolio["holdings"][selected_delete]
                try:
                    save_portfolio(PORTFOLIO_FILE, remove_holding(portfolio, selected["account"], selected["code"]))
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("本地持仓已删除。")
                    st.rerun()


def _strategy_candidate_rows(candidates):
    """把策略审计记录压缩为适合页面核对的候选表。"""
    rows = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        daily = item.get("日线证据", {}) if isinstance(item.get("日线证据"), dict) else {}
        minute = item.get("分时证据", {}) if isinstance(item.get("分时证据"), dict) else {}
        rows.append({
            "股票代码": item.get("股票代码"),
            "股票名称": item.get("股票名称"),
            "数据时间": item.get("数据时间"),
            "涨幅(%)": item.get("涨幅(%)"),
            "量比": item.get("量比"),
            "换手率(%)": item.get("换手率(%)"),
            "流通市值(亿元)": item.get("流通市值(亿元)"),
            "量能递增": daily.get("volume_staircase", "数据不足"),
            "均线多头": daily.get("ma_bull", "数据不足"),
            "全天VWAP上方": minute.get("all_at_or_above_vwap", "数据不足"),
            "14:30附近创高": minute.get("near_1430_new_high", "数据不足"),
        })
    return rows


def render_strategy_center(quant_snapshot):
    """渲染可执行策略卡和已保存筛选结果；运行策略始终需要用户明确点击。"""
    st.header("策略中心")
    st.caption("策略 skill 用于规范策略流程；页面执行的是可复现的策略模块。运行会访问公开行情，并保存带时间戳的本地筛选快照。")
    st.dataframe(strategy_catalog(), use_container_width=True, hide_index=True)
    catalog = load_catalog(CATALOG_FILE)
    if len(catalog) < 3000:
        st.warning(f"本地A股代码目录当前仅有 {len(catalog)} 条，无法完成全市场筛选。")
        if st.button("更新本地A股代码目录"):
            with st.spinner("正在更新公开A股代码目录…"):
                refresh_result = refresh_catalog()
            if refresh_result.get("status") == "success":
                st.success(f"代码目录已更新：{refresh_result.get('count')} 只。")
                st.rerun()
            else:
                st.error(refresh_result.get("message", "代码目录更新失败。"))
    if st.button("运行 A 股午后强势筛选", type="primary", disabled=len(catalog) < 3000):
        with st.spinner("正在核验全市场行情、日线和分时证据…"):
            result = run_afternoon_momentum_screen(catalog)
        if result.get("status") == "success":
            save_strategy_run(result, OUTPUT_DIRECTORY)
            st.success(result.get("message", "筛选完成。"))
            st.rerun()
        elif result.get("status") == "not_ready":
            st.info(result.get("message", "当前尚未到策略运行时间。"))
        else:
            st.error(result.get("message", "策略运行失败。"))

    result = load_latest_strategy_run(OUTPUT_DIRECTORY)
    if not result:
        st.info("尚无已保存的策略运行记录。交易日14:30后可手动运行首个策略。")
        return
    st.markdown("#### 最近一次运行")
    st.caption(
        f"数据源：{result.get('data_source', '未记录')} ｜ 获取时间：{result.get('retrieved_at', '未记录')} ｜ "
        f"范围：{result.get('universe', '未记录')} {result.get('universe_count', '未记录')} 只 ｜ "
        f"初筛：{result.get('initial_count', '未记录')} 只"
    )
    candidates = result.get("candidates", [])
    candidate_rows = _strategy_candidate_rows(candidates)
    if candidate_rows:
        st.dataframe(candidate_rows, use_container_width=True, hide_index=True)
        _render_snapshot_buy(candidates, result)
    else:
        st.info(result.get("message", "今日无完全符合条件的候选。"))
    near_misses = result.get("near_misses", [])
    if near_misses:
        st.markdown("#### 接近条件但未入选")
        st.dataframe(
            [{"股票代码": item.get("股票代码"), "股票名称": item.get("股票名称"), "未通过条件": "；".join(item.get("未通过条件", []))} for item in near_misses],
            use_container_width=True,
            hide_index=True,
        )


def _render_snapshot_buy(candidates, result):
    """仅在用户再次确认后，把候选按快照收盘价加入独立模拟账本。"""
    try:
        simulator = load_simulator(SIMULATOR_FILE)
    except ValueError as error:
        st.error(str(error))
        return
    accounts = [item["account"] for item in simulator.get("cash", [])]
    with st.expander("按快照收盘价建立模拟仓", expanded=False):
        st.caption("这不是实际委托，也不宣称按下一交易日成交；只在本地模拟账本中按本次策略快照价格记账。")
        if not accounts:
            st.info("请先在“模拟仓”录入现金余额，再建立模拟仓位。")
            return
        labels = [f"{item.get('股票名称')}（{item.get('股票代码')}）" for item in candidates]
        selected_index = st.selectbox("策略候选", range(len(candidates)), format_func=lambda index: labels[index], key="simulator_candidate")
        account = st.selectbox("模拟账户", accounts, key="simulator_account")
        quantity = st.number_input("模拟数量（100股整数倍）", min_value=100, step=100, key="simulator_quantity")
        selected = candidates[selected_index]
        st.caption(f"快照价格：{_money(selected.get('现价'))} ｜ 策略：{result.get('strategy_name', STRATEGY_ID)}")
        confirmed = st.checkbox("我确认这是本地模拟建仓，不会向券商发送任何指令。", key="simulator_buy_confirm")
        if st.button("确认建立模拟仓", disabled=not confirmed, key="simulator_buy"):
            try:
                updated, transaction = create_snapshot_buy(
                    simulator,
                    account,
                    selected,
                    quantity,
                    result.get("strategy_id", STRATEGY_ID),
                    str(result.get("retrieved_at", ""))[:10],
                )
                save_simulator(SIMULATOR_FILE, updated)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"已建立本地模拟仓：{transaction['股票名称']} {transaction['数量']} 股。")
                st.rerun()


def render_simulator(quant_snapshot):
    """渲染与真实持仓完全分离的本地模拟仓。"""
    st.header("模拟仓（本地）")
    st.caption("模拟现金、持仓与交易流水只保存于本机 data/simulator.json；不连接券商，不执行真实交易。估值使用本地最近日线收盘。")
    try:
        simulator = load_simulator(SIMULATOR_FILE)
    except ValueError as error:
        st.error(str(error))
        return
    rows, summary = summarize_simulator(simulator, _local_quote_map(quant_snapshot))
    metrics = st.columns(4)
    metrics[0].metric("模拟持仓", summary["模拟持仓数"])
    metrics[1].metric("模拟现金", _money(summary["模拟现金"]))
    metrics[2].metric("已报价模拟市值", _money(summary["已报价模拟市值"]))
    metrics[3].metric("缺少报价", summary["缺少报价数"])
    with st.expander("设置模拟现金", expanded=not simulator.get("cash")):
        with st.form("simulator_cash_form", clear_on_submit=True):
            account = st.text_input("模拟账户名称", placeholder="例如：午后策略模拟仓")
            amount = st.number_input("模拟现金余额", min_value=0.0, step=1000.0)
            if st.form_submit_button("保存模拟现金"):
                try:
                    save_simulator(SIMULATOR_FILE, upsert_simulator_cash(simulator, account, amount))
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("本地模拟现金已保存。")
                    st.rerun()
    if rows:
        st.markdown("#### 当前模拟持仓")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("尚无模拟持仓。先设置模拟现金，再从“策略中心”的候选中建立模拟仓。")
    transactions = simulator.get("transactions", [])
    if transactions:
        st.markdown("#### 模拟交易流水")
        st.dataframe(transactions, use_container_width=True, hide_index=True)


def main():
    """启动本机 Streamlit 页面。"""
    if st is None:
        raise RuntimeError("未安装 streamlit，请先执行 .venv/bin/pip install -r requirements.txt。")

    st.set_page_config(page_title="AStockAI 本地查询", page_icon="📈", layout="wide")
    st.title("AStockAI 本地 Web 查询窗口")
    st.caption("已有快照只读展示；非快照股票仅在明确点击后下载单股日线。不会自动发送邮件或运行回测；研究总览仅在显式刷新后重建本地研究产物。")
    dashboard = load_research_dashboard_cached(
        str(PROJECT_DIRECTORY), research_dashboard_source_mtime(PROJECT_DIRECTORY)
    )

    quant_file = find_latest_snapshot_file(OUTPUT_DIRECTORY, "quant_snapshot")
    if quant_file is None:
        st.warning("未找到 quant_snapshot；单股票与日报页面暂不可用，但仍可查看已有研究总览。")
        render_research_dashboard(dashboard)
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

    home_tab, portfolio_tab, strategy_tab, simulator_tab, query_tab, watchlist_tab, research_tab = st.tabs(
        ["投资总览", "持仓管理", "策略中心", "模拟仓", "单股票查询", "关注列表", "系统状态"]
    )
    with home_tab:
        render_home(quant_snapshot, watchlist_snapshot, get_report_file(report_date), daily_signal)
    with portfolio_tab:
        render_portfolio(quant_snapshot, daily_signal)
    with strategy_tab:
        render_strategy_center(quant_snapshot)
    with simulator_tab:
        render_simulator(quant_snapshot)
    with research_tab:
        render_research_dashboard(dashboard)
    with query_tab:
        render_stock_query(stock_records, daily_signal)
    with watchlist_tab:
        render_watchlist(watchlist_data)


if __name__ == "__main__":
    main()
