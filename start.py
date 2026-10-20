"""仅依赖标准库的跨平台启动器：python3 start.py / py -3 start.py。"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


PROJECT = Path(__file__).resolve().parent


def environment_python(project=PROJECT):
    return Path(project) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_environment(project=PROJECT, reinstall=False):
    project = Path(project)
    python = environment_python(project)
    if not python.is_file():
        print("正在创建项目虚拟环境……", flush=True)
        venv.EnvBuilder(with_pip=True).create(project / ".venv")
    requirements = project / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = project / ".venv" / ".astockai-requirements"
    installed = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if reinstall or installed != digest:
        print("正在安装项目依赖（首次运行需要联网）……", flush=True)
        marker.unlink(missing_ok=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements)],
            cwd=project, check=True,
        )
        marker.write_text(digest + "\n", encoding="utf-8")
    return python


def main(argv=None):
    parser = argparse.ArgumentParser(description="安装依赖并启动 AStockAI 本地工作台。")
    parser.add_argument("--install-only", action="store_true", help="只安装依赖，不启动网页。")
    parser.add_argument("--reinstall", action="store_true", help="重新检查并安装 requirements.txt 中的依赖。")
    parser.add_argument("--port", type=int, default=8501, help="本机端口，默认 8501。")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.exit(1, "请先安装 Python 3.10 或更高版本（推荐 3.12）。\n")
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间。")
    try:
        python = prepare_environment(reinstall=args.reinstall)
        if args.install_only:
            print("依赖已就绪。再次运行 start.py 即可打开工作台。")
            return 0
        print(f"AStockAI：http://127.0.0.1:{args.port}（按 Ctrl+C 停止）", flush=True)
        return subprocess.run(
            [str(python), "-m", "streamlit", "run", str(PROJECT / "app.py"),
             "--server.address=127.0.0.1", f"--server.port={args.port}"],
            cwd=PROJECT, check=False,
        ).returncode
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"启动未完成：{error}\n请检查 Python、网络和目录权限，修复后重跑 start.py。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
