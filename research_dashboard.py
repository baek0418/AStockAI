"""只读汇总本地研究产物，供 Web 研究总览展示。"""

import json
from datetime import date
from pathlib import Path


REPORT_SPECS = {
    "数据审计": ("output/prediction", "prediction_data_audit_*.json"),
    "样本外验证": ("output/prediction", "prediction_outperform_evaluation_*.json"),
    "研究对比": ("output/prediction", "prediction_comparison_*.json"),
    "组合回测": ("output/portfolio", "oos_portfolio_research_*.json"),
    "模型元数据": ("models", "predict_5d_outperform_benchmark_metadata.json"),
}


def find_latest_report(project_directory, relative_directory, pattern):
    """按文件名日期读取最新 JSON；损坏文件不能被静默当成有效研究结果。"""
    directory = Path(project_directory) / relative_directory
    files = sorted(directory.glob(pattern))
    if not files:
        return None, None, f"未找到 {directory / pattern}。"
    report_file = files[-1]
    try:
        data = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, report_file, f"无法读取 {report_file.name}：{error}。"
    if not isinstance(data, dict):
        return None, report_file, f"{report_file.name} 不是 JSON 对象。"
    return data, report_file, None


def _report(project_directory, name):
    directory, pattern = REPORT_SPECS[name]
    data, report_file, error = find_latest_report(project_directory, directory, pattern)
    return {"数据": data, "文件": str(report_file) if report_file else None, "错误": error}


def _status(data, error, ready=False):
    if error:
        return "数据不足"
    return "可用" if ready else "未通过"


def _date_lag(reference_date, data_date):
    """返回日历日滞后；日期不完整时不作猜测。"""
    try:
        return (date.fromisoformat(reference_date) - date.fromisoformat(data_date)).days
    except (TypeError, ValueError):
        return None


def _build_data_health(audit_report, reference_date):
    audit = audit_report["数据"]
    if not audit:
        return {
            "状态": "数据不足",
            "说明": audit_report["错误"],
            "报告文件": audit_report["文件"],
            "可训练股票数": None,
            "最新日线日期": None,
            "数据问题文件数": None,
            "研究池外文件数": None,
        }
    files = audit.get("文件审计", [])
    latest_dates = [
        row.get("日期范围", [None, None])[1]
        for row in files if isinstance(row, dict) and len(row.get("日期范围", [])) >= 2
    ]
    data_problem_count = sum(
        1 for row in files if isinstance(row, dict) and row.get("状态") in {"无效", "数据不足"}
    )
    excluded_count = sum(
        1 for row in files if isinstance(row, dict) and row.get("状态") == "研究池外"
    )
    latest_date = max((item for item in latest_dates if item), default=None)
    lag_days = _date_lag(reference_date, latest_date)
    freshness = (
        "未找到可比的最新量化快照日期。" if lag_days is None
        else "与最新量化快照同日。" if lag_days <= 0
        else f"相对最新量化快照滞后 {lag_days} 个日历日；重新运行审计前不要把它当作当前覆盖。"
    )
    return {
        "状态": "可用",
        "说明": audit.get("横截面限制", "未提供数据覆盖说明。"),
        "生成时间": audit.get("生成时间"),
        "报告文件": audit_report["文件"],
        "可训练股票数": audit.get("股票数量"),
        "最新日线日期": latest_date,
        "参考量化快照日期": reference_date,
        "时效滞后日": lag_days,
        "时效说明": freshness,
        "数据问题文件数": data_problem_count,
        "研究池外文件数": excluded_count,
        "文件审计": files,
        "特征构建跳过文件": audit.get("特征构建跳过文件", []),
    }


def _build_model_validation(evaluation_report, metadata_report, comparison_report, reference_date):
    evaluation = evaluation_report["数据"]
    metadata = metadata_report["数据"]
    comparison = comparison_report["数据"]
    if not evaluation:
        return {
            "状态": "数据不足",
            "说明": evaluation_report["错误"],
            "报告文件": evaluation_report["文件"],
            "技术验证通过": False,
            "允许展示概率": False,
        }
    metrics = evaluation.get("汇总样本外指标") or {}
    baseline = evaluation.get("汇总朴素概率基线") or evaluation.get("汇总永远预测上涨率基线") or {}
    windows = evaluation.get("滚动样本外验证", [])
    successful_windows = [item for item in windows if isinstance(item, dict) and item.get("status") == "success"]
    calibrated_windows = [
        item for item in successful_windows if item.get("校准", {}).get("calibrated") is True
    ]
    technical_ready = bool(metadata and metadata.get("滚动验证通过") is True)
    comparison_conclusion = (comparison or {}).get("结论", {})
    display_allowed = comparison_conclusion.get("可接入日报或 Web") is True
    if display_allowed:
        display_note = "比较报告标记为可接入；仍须人工审阅后决定任何展示范围。"
    else:
        display_note = comparison_conclusion.get(
            "结论", "研究结果保持隔离，不能展示概率或接入日报。"
        )
    training_end_date = (evaluation.get("数据范围") or {}).get("训练日期范围", [None, None])[-1]
    lag_days = _date_lag(reference_date, training_end_date)
    return {
        "状态": _status(evaluation, evaluation_report["错误"], technical_ready),
        "说明": evaluation.get("验证状态", "未提供验证状态。"),
        "生成时间": evaluation.get("生成时间"),
        "报告文件": evaluation_report["文件"],
        "模型元数据文件": metadata_report["文件"],
        "模型版本": evaluation.get("模型版本"),
        "训练截止日期": training_end_date,
        "相对量化快照滞后日": lag_days,
        "训练股票数": (evaluation.get("数据范围") or {}).get("股票数量"),
        "训练样本数": (evaluation.get("数据范围") or {}).get("样本数"),
        "样本外指标": metrics,
        "朴素概率基线": baseline,
        "完成窗口数": len(successful_windows),
        "完成校准窗口数": len(calibrated_windows),
        "技术验证通过": technical_ready,
        "允许展示概率": display_allowed,
        "展示边界": display_note,
        "风险提示": evaluation.get("风险提示", []),
    }


def _build_portfolio_backtest(portfolio_report):
    report = portfolio_report["数据"]
    if not report:
        return {
            "状态": "数据不足",
            "说明": portfolio_report["错误"],
            "报告文件": portfolio_report["文件"],
            "统计": {},
        }
    statistics = report.get("组合统计", {})
    if not isinstance(statistics, dict):
        statistics = {}
    return {
        "状态": "可用" if statistics else "数据不足",
        "说明": "严格使用滚动样本外信号，并在下一交易日执行；结果仅供研究。",
        "报告文件": portfolio_report["文件"],
        "策略": report.get("策略", "未提供"),
        "市场基准": report.get("市场基准", "未提供"),
        "统计": statistics,
        "参数": report.get("参数", {}),
        "信号覆盖诊断": report.get("信号覆盖诊断", {}),
        "跳过文件": report.get("跳过文件", []),
    }


def build_research_dashboard(project_directory):
    """生成页面需要的研究总览；严格只读既有报告。"""
    reports = {name: _report(project_directory, name) for name in REPORT_SPECS}
    quant_snapshot, _, _ = find_latest_report(project_directory, "output", "quant_snapshot_*.json")
    reference_date = (quant_snapshot or {}).get("快照日期")
    model = _build_model_validation(
        reports["样本外验证"], reports["模型元数据"], reports["研究对比"], reference_date
    )
    return {
        "数据健康": _build_data_health(reports["数据审计"], reference_date),
        "模型验证": model,
        "组合回测": _build_portfolio_backtest(reports["组合回测"]),
        "总览结论": {
            "研究结果可展示": False,
            "说明": "总览只提供本地研究产物的审计状态；不会据此生成买卖建议或自动展示单股概率。",
            "概率展示边界": model.get("展示边界"),
        },
    }


def build_research_workbench_summary(dashboard):
    """把研究产物归纳为下一步研究动作，不把回测指标转成持仓建议。"""
    health = dashboard.get("数据健康", {}) if isinstance(dashboard, dict) else {}
    model = dashboard.get("模型验证", {}) if isinstance(dashboard, dict) else {}
    portfolio = dashboard.get("组合回测", {}) if isinstance(dashboard, dict) else {}

    if health.get("状态") != "可用":
        decision = "研究资料不足"
        next_step = "先运行“后台更新本地行情”，再运行“后台重建研究产物”。"
        tone = "warning"
    elif health.get("时效滞后日") is not None and health["时效滞后日"] > 0:
        decision = "研究资料已落后于当前快照"
        next_step = "先同步本地行情并重建研究；旧审计不能代表当前数据覆盖。"
        tone = "warning"
    elif model.get("状态") == "数据不足":
        decision = "模型验证资料不足"
        next_step = "重建研究产物，生成滚动样本外验证和对比报告。"
        tone = "warning"
    elif not model.get("技术验证通过"):
        decision = "模型未通过技术验证"
        next_step = "保持模型隔离，检查数据覆盖、时间切分与样本外窗口；不要接入持仓或日报。"
        tone = "error"
    elif not model.get("允许展示概率"):
        decision = "模型仍保持研究隔离"
        next_step = "技术检查不等于可展示；先人工复核模型与基线对比结论。"
        tone = "warning"
    else:
        decision = "研究产物等待人工复核"
        next_step = "复核跨窗口稳定性、成本与敏感性后，再决定是否扩大展示范围。"
        tone = "success"

    return {
        "研究决策": decision,
        "下一步": next_step,
        "提示级别": tone,
        "数据状态": health.get("状态", "数据不足"),
        "数据时效": health.get("时效说明", "数据不足：尚未审计。"),
        "模型状态": model.get("状态", "数据不足"),
        "模型展示边界": model.get("展示边界", "数据不足：尚无模型验证。"),
        "组合回测状态": portfolio.get("状态", "数据不足"),
    }


def build_user_system_status(dashboard):
    """把研究审计翻译成普通使用者可执行的系统状态，不暴露无须处理的技术细节。"""
    health = dashboard.get("数据健康", {}) if isinstance(dashboard, dict) else {}
    model = dashboard.get("模型验证", {}) if isinstance(dashboard, dict) else {}
    data_is_current = (
        health.get("状态") == "可用"
        and (health.get("时效滞后日") is None or health.get("时效滞后日") <= 0)
    )
    if data_is_current:
        data_status = "可用"
        headline = "今天可以正常查看持仓和日报"
        detail = "本地数据与最新量化快照一致；持仓总览和日报会使用这些已保存的事实。"
        action = "无需操作。需要新数据时，可在下方手动刷新。"
        tone = "success"
    elif health.get("状态") == "可用":
        data_status = "需要更新"
        headline = "建议先更新本地数据"
        detail = health.get("时效说明", "本地研究数据较当前快照落后。")
        action = "更新完成后，再查看持仓总览和日报。"
        tone = "warning"
    else:
        data_status = "数据不足"
        headline = "需要先准备本地数据"
        detail = health.get("说明", "尚未找到可用的数据审计结果。")
        action = "请先更新本地数据；在此之前不要把研究页面当作判断依据。"
        tone = "warning"

    if model.get("技术验证通过") and model.get("允许展示概率"):
        model_status = "已通过额外检查"
        model_detail = "模型结果仍只作辅助参考，不替代持仓计划或风险控制。"
    else:
        model_status = "暂不显示模型预测"
        model_detail = "系统已自动关闭未经验证的模型预测；这不影响你使用持仓总览、日报和已有规则分析。"

    return {
        "数据状态": data_status,
        "标题": headline,
        "说明": detail,
        "建议动作": action,
        "提示级别": tone,
        "模型状态": model_status,
        "模型说明": model_detail,
    }


def research_dashboard_source_mtime(project_directory):
    """返回总览来源的最新修改时间，用于界面缓存失效。"""
    mtimes = []
    for directory, pattern in REPORT_SPECS.values():
        for report_file in (Path(project_directory) / directory).glob(pattern):
            try:
                mtimes.append(report_file.stat().st_mtime_ns)
            except OSError:
                continue
    for pattern in ("quant_snapshot.json", "quant_snapshot_*.json"):
        for snapshot_file in (Path(project_directory) / "output").glob(pattern):
            try:
                mtimes.append(snapshot_file.stat().st_mtime_ns)
            except OSError:
                continue
    return max(mtimes, default=None)
