"""AStockAI 一键运行行情、量化快照与日报流程。"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from daily_report import run_daily_report
from daily_signal import run_daily_signal
from email_report import send_daily_report
from logger import get_logger
from prediction_benchmark_data import download_benchmarks
from research_data import run_research_data
from research_summary import run_research_summary
from update_data import run_update_data
from watchlist_snapshot import run_watchlist_snapshot


STEP_STATUS_KEYS = {
    "行情更新": "update_data",
    "市场基准更新": "market_benchmark_update",
    "量化研究": "research_data",
    "关注快照": "watchlist_snapshot",
    "每日日报": "daily_report",
    "日报邮件": "email_report",
}


def create_step_record(step_name):
    """创建流水线步骤的初始记录。"""
    return {
        "name": step_name,
        "start_time": datetime.now(),
        "end_time": None,
        "status": "failed",
        "duration": 0.0,
        "output_file": None,
        "error": "",
        "result": None,
    }


def run_step(step_name, step_function, logger):
    """执行一个步骤并记录开始时间、结束时间、耗时和异常信息。"""
    step_record = create_step_record(step_name)
    logger.info("开始执行：%s", step_name)
    start_tick = time.monotonic()

    try:
        step_record["result"] = step_function()
    except Exception as error:
        step_record["error"] = str(error)
        logger.exception("%s 执行失败", step_name)
    else:
        step_record["status"] = "success"
        logger.info("执行完成：%s", step_name)

    step_record["end_time"] = datetime.now()
    step_record["duration"] = time.monotonic() - start_tick
    return step_record


def mark_step(step_record, status, message="", output_file=None):
    """补充步骤的最终状态、错误摘要和输出文件路径。"""
    step_record["status"] = status
    step_record["error"] = message
    step_record["output_file"] = output_file
    return step_record


def is_current_output_file(output_file, step_start_time):
    """检查输出文件是否在本次步骤开始后被新生成或更新。"""
    if not output_file:
        return False

    output_path = Path(output_file)
    if not output_path.exists():
        return False

    allowed_time_difference = 2
    return output_path.stat().st_mtime >= step_start_time.timestamp() - allowed_time_difference


def count_valid_history_files(data_directory):
    """统计 data 文件夹中可供研究模块使用的历史 CSV 数量。"""
    required_columns = {"日期", "收盘", "成交量"}
    valid_count = 0

    for data_file in data_directory.glob("*.csv"):
        try:
            with open(data_file, "r", encoding="utf-8-sig", newline="") as file:
                header = next(csv.reader(file), [])
        except OSError:
            continue

        if required_columns.issubset(set(header)):
            valid_count += 1

    return valid_count


def get_update_status(step_record, data_directory, logger):
    """根据更新模块的结构化结果判断行情更新步骤状态。"""
    update_result = step_record["result"]
    valid_data_count = count_valid_history_files(data_directory)

    if not isinstance(update_result, dict):
        return mark_step(step_record, "failed", "行情更新模块未返回有效结果。")

    details = update_result.get("details", {})
    success_count = details.get("success_count", 0)
    failed_count = details.get("failed_count", 0)
    step_record["details"] = {
        "success_count": success_count,
        "failed_count": failed_count,
        "valid_data_count": valid_data_count,
    }

    if update_result.get("status") == "success":
        return mark_step(step_record, "success")

    if valid_data_count > 0:
        message = update_result.get("message", "行情更新存在失败，使用现有有效行情继续。")
        logger.warning("%s，有效历史 CSV：%s", message, valid_data_count)
        return mark_step(step_record, "partial", message)

    return mark_step(step_record, "failed", update_result.get("message", "行情更新失败。"))


def run_market_benchmark_update():
    """更新日报依赖的两个市场基准，并返回与个股更新一致的结构化结果。"""
    results = download_benchmarks()
    success_count = sum(item.get("status") == "success" for item in results)
    failed_count = len(results) - success_count
    status = "success" if failed_count == 0 else "partial" if success_count else "failed"
    return {
        "status": status,
        "message": "市场基准更新完成" if not failed_count else "部分市场基准更新失败",
        "details": {"success_count": success_count, "failed_count": failed_count, "results": results},
    }


def get_market_benchmark_update_status(step_record, logger):
    """记录指数更新结果；失败时允许日报继续明确展示已有基准日期。"""
    result = step_record["result"]
    if not isinstance(result, dict):
        return mark_step(step_record, "failed", "市场基准更新模块未返回有效结果。")
    details = result.get("details", {})
    step_record["details"] = {
        "success_count": details.get("success_count", 0),
        "failed_count": details.get("failed_count", 0),
    }
    status = result.get("status")
    if status == "success":
        return mark_step(step_record, "success")
    message = result.get("message", "市场基准更新失败，日报将明确保留已有基准数据日期。")
    if status == "partial":
        logger.warning(message)
        return mark_step(step_record, "partial", message)
    logger.warning(message)
    return mark_step(step_record, "failed", message)


def get_snapshot_status(step_record, logger):
    """校验研究或关注快照步骤返回的本次输出文件。"""
    step_result = step_record["result"]

    if not isinstance(step_result, tuple) or len(step_result) < 2:
        return mark_step(step_record, "failed", "模块未返回本次生成的快照文件。")

    snapshot_data, output_file = step_result[0], step_result[1]
    if not is_current_output_file(output_file, step_record["start_time"]):
        return mark_step(step_record, "failed", "未确认本次生成新的快照文件。")

    if not isinstance(snapshot_data, dict):
        return mark_step(step_record, "failed", "快照内容格式异常。")

    output_path = str(output_file)
    if "matched" in snapshot_data and "missing" in snapshot_data:
        matched_count = snapshot_data["matched"]
        missing_count = snapshot_data["missing"]
        step_record["details"] = {
            "matched": matched_count,
            "missing": missing_count,
        }
        status = "partial" if missing_count else "success"
        return mark_step(step_record, status, "" if not missing_count else "部分关注股未匹配。", output_path)

    return mark_step(step_record, "success", "", output_path)


def run_daily_report_step():
    """先同步摘要和日间信号，再生成依赖两者的每日关注股票日报。"""
    _, summary_file = run_research_summary()
    signal_file = run_daily_signal()
    report_file = run_daily_report()
    return {
        "summary_file": summary_file,
        "signal_file": signal_file,
        "output_file": report_file,
    }


def get_daily_report_status(step_record):
    """校验日报是否为本次生成，并识别 AI 降级状态。"""
    step_result = step_record["result"]
    if not isinstance(step_result, dict):
        return mark_step(step_record, "failed", "日报模块未返回有效结果。")

    report_file = step_result.get("output_file")
    if not is_current_output_file(report_file, step_record["start_time"]):
        return mark_step(step_record, "failed", "未确认本次生成新的日报文件。")

    try:
        report_content = Path(report_file).read_text(encoding="utf-8")
    except OSError as error:
        return mark_step(step_record, "failed", f"无法读取日报文件：{error}")

    ai_unavailable = "AI增强分析暂不可用" in report_content
    step_record["details"] = {"ai_unavailable": ai_unavailable}
    status = "partial" if ai_unavailable else "success"
    message = "AI增强分析不可用，已保留规则分析。" if ai_unavailable else ""
    return mark_step(step_record, status, message, str(report_file))


def get_email_report_status(step_record, report_file, logger):
    """根据邮件模块的结构化结果记录发送状态，不暴露 SMTP 敏感配置。"""
    email_result = step_record["result"]
    if not isinstance(email_result, dict):
        return mark_step(step_record, "failed", "邮件模块未返回有效结果。", str(report_file))

    message = email_result.get("message", "邮件发送失败。")
    if email_result.get("status") == "success":
        return mark_step(step_record, "success", "", str(report_file))

    logger.error("日报邮件发送失败：%s", message)
    return mark_step(step_record, "failed", message, str(report_file))


def create_skipped_step(step_name, message):
    """创建未执行步骤的记录，确保终端摘要完整。"""
    now = datetime.now()
    return {
        "name": step_name,
        "start_time": now,
        "end_time": now,
        "status": "skipped",
        "duration": 0.0,
        "output_file": None,
        "error": message,
        "result": None,
    }


def serialize_step_record(step_record):
    """将运行时步骤记录转换为可安全保存的 JSON 数据。"""
    return {
        "status": step_record["status"],
        "started_at": step_record["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": step_record["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "duration": round(step_record["duration"], 2),
        "output_file": str(step_record["output_file"]) if step_record["output_file"] else None,
        "error": step_record["error"],
        "details": step_record.get("details", {}),
    }


def get_pipeline_status(pipeline_result):
    """根据真实步骤状态汇总本次流水线的整体状态。"""
    if not pipeline_result["success"]:
        return "failed"

    statuses = [step["status"] for step in pipeline_result["steps"]]
    if any(status in {"partial", "skipped"} for status in statuses):
        return "partial"

    return "success"


def save_pipeline_status(pipeline_result):
    """使用临时文件和原子替换保存本次流水线状态。"""
    project_directory = Path(__file__).parent
    output_directory = project_directory / "output"
    output_directory.mkdir(exist_ok=True)
    status_file = output_directory / "pipeline_status.json"
    temporary_file = output_directory / "pipeline_status.json.tmp"
    step_statuses = {
        STEP_STATUS_KEYS[step["name"]]: serialize_step_record(step)
        for step in pipeline_result["steps"]
        if step["name"] in STEP_STATUS_KEYS
    }
    status_data = {
        "run_id": pipeline_result["run_id"],
        "started_at": pipeline_result["started_at"],
        "finished_at": pipeline_result["finished_at"],
        "status": pipeline_result["status"],
        "duration": round(pipeline_result["duration"], 2),
        "steps": step_statuses,
    }

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(status_data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_file, status_file)
    return status_file


def finish_pipeline_result(pipeline_result, pipeline_started_at):
    """补全流水线元数据、保存状态文件并输出终端摘要。"""
    pipeline_result["started_at"] = pipeline_started_at.strftime("%Y-%m-%d %H:%M:%S")
    pipeline_result["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_result["run_id"] = pipeline_started_at.strftime("%Y%m%d_%H%M%S")
    pipeline_result["status"] = get_pipeline_status(pipeline_result)
    pipeline_result["status_file"] = save_pipeline_status(pipeline_result)
    print_pipeline_summary(pipeline_result)
    return pipeline_result


def run_pipeline(skip_update=False, no_ai=False, continue_on_error=False, send_email=False):
    """按固定顺序执行完整流水线，并返回每个步骤的真实结果。"""
    project_directory = Path(__file__).parent
    data_directory = project_directory / "data"
    logger = get_logger("pipeline")
    pipeline_start = time.monotonic()
    pipeline_started_at = datetime.now()
    steps = []

    if no_ai:
        os.environ["ASTOCKAI_DISABLE_AI"] = "1"
        logger.info("本次流水线已禁用 AI 增强。")

    logger.info("AStockAI Pipeline 开始：%s", pipeline_started_at.strftime("%Y-%m-%d %H:%M:%S"))

    if skip_update:
        update_step = create_skipped_step("行情更新", "已跳过行情更新，使用已有行情数据。")
        update_step["details"] = {"valid_data_count": count_valid_history_files(data_directory)}
        steps.append(update_step)
        market_step = create_skipped_step("市场基准更新", "已跳过市场基准更新，使用已有市场数据。")
        steps.append(market_step)
        logger.warning("已跳过行情更新，使用已有行情数据。")
    else:
        update_step = run_step("行情更新", run_update_data, logger)
        update_step = get_update_status(update_step, data_directory, logger)
        steps.append(update_step)
        market_step = run_step("市场基准更新", run_market_benchmark_update, logger)
        market_step = get_market_benchmark_update_status(market_step, logger)
        steps.append(market_step)

    valid_data_count = count_valid_history_files(data_directory)
    if valid_data_count == 0:
        steps.extend([
            create_skipped_step("量化研究", "没有有效历史 CSV，无法生成量化快照。"),
            create_skipped_step("关注快照", "量化研究未执行。"),
            create_skipped_step("每日日报", "量化研究未执行。"),
        ])
        if send_email:
            steps.append(create_skipped_step("日报邮件", "日报未成功生成，未发送邮件。"))
        pipeline_result = {
            "steps": steps,
            "duration": time.monotonic() - pipeline_start,
            "success": False,
        }
        return finish_pipeline_result(pipeline_result, pipeline_started_at)

    research_step = run_step("量化研究", run_research_data, logger)
    research_step = get_snapshot_status(research_step, logger)
    steps.append(research_step)

    if research_step["status"] == "failed":
        logger.error("量化研究失败，跳过关注快照和每日日报，避免误用历史快照。")
        steps.extend(
            [
                create_skipped_step("关注快照", "量化研究失败，避免误用历史快照。"),
                create_skipped_step("每日日报", "量化研究失败，避免误用历史摘要。"),
            ]
        )
        if send_email:
            steps.append(create_skipped_step("日报邮件", "日报未成功生成，未发送邮件。"))
        pipeline_result = {
            "steps": steps,
            "duration": time.monotonic() - pipeline_start,
            "success": False,
        }
        return finish_pipeline_result(pipeline_result, pipeline_started_at)

    watchlist_step = run_step("关注快照", run_watchlist_snapshot, logger)
    watchlist_step = get_snapshot_status(watchlist_step, logger)
    steps.append(watchlist_step)

    if watchlist_step["status"] == "failed" and not continue_on_error:
        logger.error("关注快照失败，默认跳过日报；可使用 --continue-on-error 明确承担风险后继续。")
        steps.append(
            create_skipped_step(
                "每日日报",
                "关注快照失败，默认不生成日报以避免不完整结果。",
            )
        )
        if send_email:
            steps.append(create_skipped_step("日报邮件", "日报未成功生成，未发送邮件。"))
        pipeline_result = {
            "steps": steps,
            "duration": time.monotonic() - pipeline_start,
            "success": False,
        }
        return finish_pipeline_result(pipeline_result, pipeline_started_at)

    if watchlist_step["status"] == "failed":
        logger.warning("关注快照失败，已根据 --continue-on-error 继续生成日报。")

    daily_step = run_step("每日日报", run_daily_report_step, logger)
    daily_step = get_daily_report_status(daily_step)
    steps.append(daily_step)

    email_succeeded = True
    if send_email:
        if daily_step["status"] in {"success", "partial"}:
            email_step = run_step(
                "日报邮件",
                lambda: send_daily_report(daily_step["output_file"]),
                logger,
            )
            email_step = get_email_report_status(
                email_step, daily_step["output_file"], logger
            )
            steps.append(email_step)
            email_succeeded = email_step["status"] == "success"
        else:
            steps.append(create_skipped_step("日报邮件", "日报未成功生成，未发送邮件。"))

    pipeline_result = {
        "steps": steps,
        "duration": time.monotonic() - pipeline_start,
        "success": daily_step["status"] in {"success", "partial"} and email_succeeded,
    }
    return finish_pipeline_result(pipeline_result, pipeline_started_at)


def get_status_text(status):
    """将内部步骤状态转换为终端可读中文。"""
    status_texts = {
        "success": "成功",
        "partial": "部分成功",
        "failed": "失败",
        "skipped": "已跳过",
    }
    return status_texts.get(status, status)


def format_output_path(output_file):
    """将输出路径转换为相对项目目录的显示形式。"""
    if not output_file:
        return "无"

    project_directory = Path(__file__).parent
    output_path = Path(output_file)

    try:
        return str(output_path.relative_to(project_directory))
    except ValueError:
        return str(output_path)


def print_pipeline_summary(pipeline_result):
    """根据真实步骤结果输出流水线终端摘要。"""
    step_lookup = {step["name"]: step for step in pipeline_result["steps"]}
    update_step = step_lookup.get("行情更新")
    market_step = step_lookup.get("市场基准更新")
    research_step = step_lookup.get("量化研究")
    watchlist_step = step_lookup.get("关注快照")
    daily_step = step_lookup.get("每日日报")
    email_step = step_lookup.get("日报邮件")

    print("========================================")
    print("AStockAI Pipeline 执行完成")
    print()

    if update_step:
        print(f"行情更新：{get_status_text(update_step['status'])}")
        details = update_step.get("details", {})
        if update_step["status"] == "skipped":
            print("说明：已跳过行情更新，使用已有行情数据")
        elif details:
            print(f"成功股票：{details.get('success_count', 0)}")
            print(f"失败股票：{details.get('failed_count', 0)}")

    if market_step:
        print()
        print(f"市场基准更新：{get_status_text(market_step['status'])}")
        details = market_step.get("details", {})
        if details:
            print(f"成功指数：{details.get('success_count', 0)}")
            print(f"失败指数：{details.get('failed_count', 0)}")

    if research_step:
        print()
        print(f"量化研究：{get_status_text(research_step['status'])}")
        print(f"量化快照：{format_output_path(research_step['output_file'])}")

    if watchlist_step:
        print()
        print(f"关注快照：{get_status_text(watchlist_step['status'])}")
        details = watchlist_step.get("details", {})
        if details:
            print(f"匹配：{details.get('matched', 0)}")
            print(f"缺失：{details.get('missing', 0)}")
        print(f"关注快照文件：{format_output_path(watchlist_step['output_file'])}")

    if daily_step:
        print()
        print(f"每日日报：{get_status_text(daily_step['status'])}")
        print(f"报告：{format_output_path(daily_step['output_file'])}")
        ai_unavailable = daily_step.get("details", {}).get("ai_unavailable")
        if ai_unavailable:
            print("AI状态：未配置或不可用，已使用规则分析")

    print()
    if email_step:
        print(f"日报邮件：{get_status_text(email_step['status'])}")
        if email_step["error"]:
            print(f"说明：{email_step['error']}")
    else:
        print("日报邮件：未请求（默认不发送）")

    print()
    print(f"总耗时：{pipeline_result['duration']:.1f} 秒")
    log_file = Path(__file__).parent / "logs" / f"astockai_{datetime.now().strftime('%Y-%m-%d')}.log"
    print(f"日志：{format_output_path(log_file)}")
    print("========================================")


def main():
    """解析命令行参数并返回适合 shell 使用的退出码。"""
    parser = argparse.ArgumentParser(description="AStockAI 一键运行数据、量化和日报流程。")
    parser.add_argument("--skip-update", action="store_true", help="跳过行情更新，使用已有行情数据。")
    parser.add_argument("--no-ai", action="store_true", help="仅本次运行禁用 AI 增强。")
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="日报成功生成后，通过 .env 配置的 QQ 邮箱发送邮件。",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="在数据一致性允许时继续后续步骤。",
    )
    arguments = parser.parse_args()
    pipeline_result = run_pipeline(
        skip_update=arguments.skip_update,
        no_ai=arguments.no_ai,
        continue_on_error=arguments.continue_on_error,
        send_email=arguments.send_email,
    )

    return 0 if pipeline_result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
