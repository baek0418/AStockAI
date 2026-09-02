"""本机后台任务的启动、状态持久化和执行入口。"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from astock_core.data.prediction_benchmark_data import download_benchmarks
from astock_core.research.research_refresh import refresh_research_artifacts
from astock_core.data.update_data import run_update_data


TASK_TYPES = {"data_update", "research_refresh"}
ACTIVE_STATUSES = {"queued", "running"}


def task_directory(project_directory):
    return Path(project_directory) / "output" / "tasks"


def task_status_file(project_directory, task_id):
    return task_directory(project_directory) / f"{task_id}.json"


def task_log_file(project_directory, task_id):
    return task_directory(project_directory) / f"{task_id}.log"


def _write_json_atomically(file_path, data):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_task_status(status_file):
    try:
        data = json.loads(Path(status_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def update_task_status(project_directory, task_id, **changes):
    """原子更新任务状态，保留已写入的步骤和进程信息。"""
    file_path = task_status_file(project_directory, task_id)
    current = load_task_status(file_path) or {"任务编号": task_id}
    current.update(changes)
    current["更新时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_json_atomically(file_path, current)
    return current


def list_task_statuses(project_directory, limit=8):
    """读取最近任务状态；单个损坏状态文件不阻断页面。"""
    statuses = []
    for status_file in task_directory(project_directory).glob("*.json"):
        status = load_task_status(status_file)
        if status:
            statuses.append(status)
    return sorted(statuses, key=lambda item: item.get("开始时间", ""), reverse=True)[:limit]


def _pid_is_running(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def get_active_task(project_directory):
    """返回活跃任务；若进程已异常消失，明确标为失败并解除阻塞。"""
    for task in list_task_statuses(project_directory):
        if task.get("状态") not in ACTIVE_STATUSES:
            continue
        if task.get("状态") == "queued" and task.get("进程ID") is None:
            try:
                queued_at = datetime.strptime(task["开始时间"], "%Y-%m-%d %H:%M:%S")
                queued_seconds = (datetime.now() - queued_at).total_seconds()
            except (KeyError, TypeError, ValueError):
                queued_seconds = 0
            if queued_seconds < 60:
                return task
            return update_task_status(
                project_directory,
                task["任务编号"],
                状态="failed",
                结束时间=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                说明="任务排队超过 60 秒仍未启动，已安全标记为失败。",
            )
        if _pid_is_running(task.get("进程ID")):
            return task
        return update_task_status(
            project_directory,
            task["任务编号"],
            状态="failed",
            结束时间=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            说明="后台进程已退出，但未写入完成状态。请查看任务日志后重试。",
        )
    return None


def start_background_task(task_type, project_directory, process_factory=subprocess.Popen):
    """启动一个独立 Python 进程；同一时间只允许一个写入本地研究数据的任务。"""
    if task_type not in TASK_TYPES:
        return {"status": "failed", "message": "未知后台任务类型。"}
    project_directory = Path(project_directory).resolve()
    active_task = get_active_task(project_directory)
    if active_task:
        return {
            "status": "busy",
            "message": f"已有后台任务“{active_task.get('任务类型', '未知')}”正在运行。",
            "task": active_task,
        }
    task_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    initial = {
        "任务编号": task_id,
        "任务类型": task_type,
        "状态": "queued",
        "开始时间": started_at,
        "结束时间": None,
        "进程ID": None,
        "说明": "任务已排队，等待后台进程启动。",
        "步骤": [],
        "日志文件": str(task_log_file(project_directory, task_id)),
    }
    _write_json_atomically(task_status_file(project_directory, task_id), initial)
    python_executable = project_directory / ".venv" / "bin" / "python"
    executable = str(python_executable) if python_executable.is_file() else sys.executable
    command = [executable, "background_task_runner.py", task_type, task_id]
    try:
        with task_log_file(project_directory, task_id).open("a", encoding="utf-8") as log_file:
            process = process_factory(
                command,
                cwd=project_directory,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as error:
        update_task_status(
            project_directory, task_id, 状态="failed", 结束时间=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            说明=f"无法启动后台进程：{error}",
        )
        return {"status": "failed", "message": "无法启动后台任务。", "task_id": task_id}
    task = update_task_status(project_directory, task_id, 进程ID=process.pid, 说明="后台进程已启动。")
    return {"status": "started", "message": "后台任务已启动，可在此页面查看进度。", "task": task}


def _run_step(name, function):
    started = time.monotonic()
    try:
        result = function()
    except Exception as error:
        return {"步骤": name, "状态": "failed", "说明": str(error), "耗时秒": round(time.monotonic() - started, 2)}
    if isinstance(result, dict):
        status = result.get("status", "success")
        message = result.get("message", "完成。")
    else:
        status, message = "success", "完成。"
    return {"步骤": name, "状态": status, "说明": message, "耗时秒": round(time.monotonic() - started, 2)}


def _overall_status(steps):
    statuses = {step["状态"] for step in steps}
    if statuses == {"success"}:
        return "success"
    if statuses == {"failed"}:
        return "failed"
    return "partial"


def _update_benchmarks():
    results = download_benchmarks()
    success_count = sum(item.get("status") == "success" for item in results)
    failed_count = len(results) - success_count
    if success_count == len(results):
        status, message = "success", "市场基准更新完成。"
    elif success_count:
        status, message = "partial", "部分市场基准更新失败。"
    else:
        status, message = "failed", "市场基准更新失败。"
    return {"status": status, "message": message, "details": {"results": results, "failed_count": failed_count}}


def run_background_task(task_type, task_id, project_directory):
    """由独立进程调用，并持续把可读进度写回状态文件。"""
    project_directory = Path(project_directory).resolve()
    if task_type not in TASK_TYPES:
        raise ValueError("未知后台任务类型。")
    update_task_status(project_directory, task_id, 状态="running", 说明="任务正在运行。")
    try:
        if task_type == "data_update":
            update_task_status(
                project_directory, task_id, 步骤=[{"步骤": "股票日线更新", "状态": "running", "说明": "正在请求股票日线。"}]
            )
            stock_step = _run_step("股票日线更新", run_update_data)
            update_task_status(
                project_directory,
                task_id,
                步骤=[stock_step, {"步骤": "市场基准更新", "状态": "running", "说明": "正在请求市场指数日线。"}],
            )
            benchmark_step = _run_step("市场基准更新", _update_benchmarks)
            steps = [stock_step, benchmark_step]
            boundary = "已显式请求网络更新；只更新本地 data/ 与 data/market/，不训练模型、不发送邮件、不生成日报。"
        else:
            result = refresh_research_artifacts(
                project_directory,
                step_callback=lambda completed: update_task_status(
                    project_directory,
                    task_id,
                    步骤=completed,
                    说明=f"已完成 {len(completed)}/4 个研究步骤。",
                ),
            )
            steps = result["步骤"]
            boundary = result["边界"]
        for index in range(1, len(steps) + 1):
            update_task_status(project_directory, task_id, 步骤=steps[:index], 说明=f"已完成 {index}/{len(steps)} 个步骤。")
        status = _overall_status(steps)
        return update_task_status(
            project_directory,
            task_id,
            状态=status,
            结束时间=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            说明="后台任务完成。" if status == "success" else "后台任务结束，部分步骤未完全成功。",
            步骤=steps,
            边界=boundary,
        )
    except Exception as error:
        return update_task_status(
            project_directory,
            task_id,
            状态="failed",
            结束时间=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            说明=f"后台任务异常：{error}",
        )


def read_task_log_tail(project_directory, task, max_characters=12000):
    """读取有限长度的末尾日志，避免 Web 页面加载大日志文件。"""
    log_file = (task or {}).get("日志文件")
    if not log_file:
        return ""
    try:
        content = Path(log_file).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"无法读取任务日志：{error}"
    return content[-max_characters:]
