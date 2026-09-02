"""后台任务子进程入口。"""

import argparse
from pathlib import Path

from astock_core.runtime.background_tasks import TASK_TYPES, run_background_task


def main():
    parser = argparse.ArgumentParser(description="AStockAI 本机后台任务执行器")
    parser.add_argument("task_type", choices=sorted(TASK_TYPES))
    parser.add_argument("task_id")
    arguments = parser.parse_args()
    result = run_background_task(arguments.task_type, arguments.task_id, Path(__file__).parents[1])
    print(f"任务 {arguments.task_id} 结束：{result['状态']}。")
    return 0 if result["状态"] in {"success", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
