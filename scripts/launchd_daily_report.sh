#!/bin/zsh
# 安装、停止和卸载 AStockAI 的当前用户 launchd 日报任务。

set -eu

SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIRECTORY="$(cd "$SCRIPT_DIRECTORY/.." && pwd)"
LABEL="com.astockai.daily-report"
USER_ID="$(id -u)"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"
PYTHON_EXECUTABLE="$PROJECT_DIRECTORY/.venv/bin/python"
RUNNER="$PROJECT_DIRECTORY/run_scheduled_daily_report.py"
LOGS_DIRECTORY="$PROJECT_DIRECTORY/logs"

print_usage() {
  print "用法: $0 {install|test|run-now|status|stop|start|uninstall|print-plist}"
}

require_project_files() {
  if [[ ! -x "$PYTHON_EXECUTABLE" || ! -f "$RUNNER" || ! -f "$PROJECT_DIRECTORY/.env" ]]; then
    print -u2 "错误：请确认项目 .venv、run_scheduled_daily_report.py 和根目录 .env 均存在。"
    exit 1
  fi
}

warn_if_not_beijing_offset() {
  if [[ "$(date +%z)" != "+0800" ]]; then
    print -u2 "警告：launchd 按 Mac 本地时区安排时间；当前不为 UTC+08:00。请将 Mac 时区设为中国标准时间，才能在北京时间 09:30 执行。"
  fi
}

write_plist() {
  "$PYTHON_EXECUTABLE" - "$PROJECT_DIRECTORY" "$PYTHON_EXECUTABLE" "$RUNNER" "$LOGS_DIRECTORY" "$PLIST_PATH" <<'PY'
import sys
from pathlib import Path
from xml.sax.saxutils import escape

project, python, runner, logs, plist = map(Path, sys.argv[1:])
xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.astockai.daily-report</string>
  <key>ProgramArguments</key>
  <array>
    <string>{escape(str(python))}</string>
    <string>{escape(str(runner))}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{escape(str(project))}</string>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>30</integer></dict>
  </array>
  <key>StandardOutPath</key>
  <string>{escape(str(logs / "launchd_daily_report.out.log"))}</string>
  <key>StandardErrorPath</key>
  <string>{escape(str(logs / "launchd_daily_report.err.log"))}</string>
</dict>
</plist>
'''
Path(plist).write_text(xml, encoding="utf-8")
PY
  chmod 644 "$PLIST_PATH"
}

case "${1:-}" in
  install)
    require_project_files
    warn_if_not_beijing_offset
    mkdir -p "$HOME/Library/LaunchAgents" "$LOGS_DIRECTORY"
    if launchctl print "gui/$USER_ID/$LABEL" >/dev/null 2>&1; then
      launchctl bootout "gui/$USER_ID" "$PLIST_PATH"
    fi
    write_plist
    launchctl bootstrap "gui/$USER_ID" "$PLIST_PATH"
    print "已安装：$LABEL（北京时间工作日 09:30；节假日由项目交易日历跳过）。"
    ;;
  test)
    require_project_files
    exec "$PYTHON_EXECUTABLE" "$RUNNER" --dry-run
    ;;
  run-now)
    require_project_files
    print "即将强制执行一次真实流水线并可能发送邮件。"
    exec "$PYTHON_EXECUTABLE" "$RUNNER" --force
    ;;
  status)
    launchctl print "gui/$USER_ID/$LABEL"
    ;;
  stop)
    launchctl bootout "gui/$USER_ID" "$PLIST_PATH"
    print "已停止任务；配置文件仍保留，可用 start 恢复。"
    ;;
  start)
    launchctl bootstrap "gui/$USER_ID" "$PLIST_PATH"
    print "已启动任务。"
    ;;
  uninstall)
    if launchctl print "gui/$USER_ID/$LABEL" >/dev/null 2>&1; then
      launchctl bootout "gui/$USER_ID" "$PLIST_PATH"
    fi
    rm -f "$PLIST_PATH"
    print "已卸载 $LABEL；项目日志和日报文件未删除。"
    ;;
  print-plist)
    require_project_files
    warn_if_not_beijing_offset
    print "install 会生成：$PLIST_PATH"
    print "运行命令：$PYTHON_EXECUTABLE $RUNNER"
    ;;
  *)
    print_usage
    exit 1
    ;;
esac
