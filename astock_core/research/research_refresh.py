"""显式重建本地研究产物，不下载行情也不接入日报。"""

import time
from pathlib import Path

from astock_core.research.oos_portfolio_research import run_oos_portfolio_research
from astock_core.research.prediction_comparison import create_comparison
from astock_core.research.prediction_data_audit import run_audit
from astock_core.research.train_prediction_v2 import train_prediction_v2


def _run_step(name, function):
    started = time.monotonic()
    try:
        result = function()
    except Exception as error:  # 页面入口必须把异常转为可读状态，不能中断整个总览。
        return {"步骤": name, "状态": "failed", "说明": str(error), "耗时秒": round(time.monotonic() - started, 2)}
    if isinstance(result, dict):
        raw_status = result.get("status", "success")
        status = raw_status if raw_status in {"success", "insufficient"} else "failed"
        message = result.get("message", "完成。")
    else:
        status, message = "success", "完成。"
    return {"步骤": name, "状态": status, "说明": message, "耗时秒": round(time.monotonic() - started, 2)}


def refresh_research_artifacts(project_directory, step_callback=None):
    """显式重建研究报告，始终只使用当前已有的本地历史数据。

    训练或回测数据不足时仍继续生成可用的审计、对比结果，并将不足原因返回页面。
    """
    project_directory = Path(project_directory)
    results = []
    for name, function in (
        ("预测数据审计", lambda: run_audit(project_directory)),
        ("v5.1 训练与滚动验证", lambda: train_prediction_v2(project_directory)),
        ("v5.0/v5.1 研究对比", lambda: create_comparison(project_directory)),
        ("严格样本外组合回测", lambda: run_oos_portfolio_research(project_directory)),
    ):
        results.append(_run_step(name, function))
        if step_callback:
            try:
                step_callback(list(results))
            except Exception:
                # 状态展示不能反过来阻断本地研究流程。
                pass
    statuses = {item["状态"] for item in results}
    overall = "failed" if "failed" in statuses else "insufficient" if "insufficient" in statuses else "success"
    return {
        "状态": overall,
        "说明": "研究产物刷新完成。" if overall == "success" else "研究产物已刷新，但部分步骤未形成可用结果。",
        "步骤": results,
        "边界": "本次只读取本地 data/ 与既有研究配置；不下载行情、不发送邮件、不修改日报。",
    }
