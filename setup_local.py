"""首次使用 AStockAI 的本地初始化入口。"""

import json
from pathlib import Path

from pipeline import run_pipeline


def ensure_local_watchlist(project_directory):
    """只在本地关注列表缺失时，从不含个人信息的模板创建空配置。"""
    project = Path(project_directory)
    watchlist_file = project / "watchlist.json"
    template_file = project / "watchlist.example.json"
    if watchlist_file.exists():
        return {"created": False, "path": watchlist_file}
    try:
        template = json.loads(template_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取关注列表模板：{error}。") from error
    if not isinstance(template, dict) or not isinstance(template.get("stocks"), list):
        raise ValueError("关注列表模板格式错误。")
    watchlist_file.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"created": True, "path": watchlist_file}


def run_first_setup(project_directory=None, pipeline_runner=run_pipeline):
    """创建本地空关注列表，再显式下载首批数据和快照；不启用 AI 或邮件。"""
    project = Path(project_directory or Path(__file__).parent)
    watchlist = ensure_local_watchlist(project)
    result = pipeline_runner(no_ai=True, send_email=False)
    return {
        "success": bool(result.get("success")) if isinstance(result, dict) else False,
        "watchlist_created": watchlist["created"],
        "watchlist_file": str(watchlist["path"]),
        "pipeline": result,
    }


def main():
    result = run_first_setup()
    print("本地关注列表：" + ("已创建" if result["watchlist_created"] else "已存在"))
    print("首次数据初始化：" + ("完成" if result["success"] else "未完成；请查看上方流水线摘要。"))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
