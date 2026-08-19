# AStockAI macOS 自动日报

`scripts/launchd_daily_report.sh` 只在你执行 `install` 时创建当前用户的 launchd 配置；仓库本身不会安装、修改或删除任何系统任务。

自动任务在 Mac 本地时区为中国标准时间（UTC+08:00）时，于每个工作日 09:30 启动。`run_scheduled_daily_report.py` 会使用北京时间再次校验 A 股交易日：周末、配置中的交易所休市日以及未配置年度都会跳过，不更新数据也不发送邮件。

## 安装

先确认项目根目录已有 `.env`，且 `.venv` 已创建，然后运行：

```bash
scripts/launchd_daily_report.sh install
```

任务会安装在 `~/Library/LaunchAgents/com.astockai.daily-report.plist`，标准输出和错误输出分别写入项目 `logs/launchd_daily_report.out.log` 与 `logs/launchd_daily_report.err.log`。定时任务本身的交易日判断日志写入 `logs/scheduled_daily_report_YYYY-MM-DD.log`。

## 测试与管理

```bash
# 不更新数据、不发邮件，只检查 .env、日历和当前交易日判断
scripts/launchd_daily_report.sh test

# 立即执行一次真实流水线（忽略交易日判断，可能发送邮件）
scripts/launchd_daily_report.sh run-now

# 查看状态
scripts/launchd_daily_report.sh status

# 停止但保留配置；可用 start 恢复
scripts/launchd_daily_report.sh stop
scripts/launchd_daily_report.sh start

# 卸载 launchd 配置；不会删除项目日志或日报
scripts/launchd_daily_report.sh uninstall
```

## 维护与睡眠说明

`config/a_share_market_holidays.json` 当前包含 2026 年上交所休市日。每年交易所发布新年度安排后，应先更新该文件；若日历没有当年配置，任务会安全跳过并记录错误，避免在不确定日期发送邮件。

launchd 的 `StartCalendarInterval` 按 Mac 本地时区运行，因此 Mac 时区必须保持为中国标准时间，才能对应北京时间 09:30。Mac 睡眠、断电或关机时，任务可能无法在准确时刻执行（唤醒后的补跑也不应视为准点保证）；恢复后请查看日志，必要时手动执行 `scripts/launchd_daily_report.sh run-now`。
