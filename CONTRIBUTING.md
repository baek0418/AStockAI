# 开发与贡献

## 本地开发

推荐 Python 3.12。克隆后运行 `python3 start.py --install-only`（Windows 使用 `py -3`），然后运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

Windows 将 Python 路径替换为 `.venv\Scripts\python.exe`。测试使用临时目录和离线替身，不需要行情账户、AI Key 或邮件配置。GitHub Actions 在 Linux / Windows 和 Python 3.10–3.12 上验证。

## 代码放在哪里

| 目录 / 入口 | 职责 |
|---|---|
| `start.py` / `setup_local.py` | 安装启动 / 显式初始化行情、快照和日报 |
| `app.py` | Streamlit 页面与交互 |
| `astock_core/data/` | 行情适配、股票池、前复权和日期校验 |
| `astock_core/analysis/` / `strategies/` | 证据核对、规则分析和策略框架 |
| `astock_core/portfolio/` / `simulator/` | 手动持仓和独立模拟仓 |
| `astock_core/reporting/` / `research/` | 日报、邮件及研究验证 |
| `astock_core/runtime/` | 后台任务、日志与状态 |
| `config/` / `docs/` / `tests/` | 公开配置、详细说明和测试 |
| `scripts/` | 高级研究和恢复工具，不参与首次启动 |

## 提交要求

- 阅读 [AGENTS.md](AGENTS.md) 和 [数据说明](docs/data_sources.md)。禁止请求任何东方财富接口，包括通过包装库间接调用及失败后的备用请求。
- 正式日线只接受前复权；失败保留旧文件并记录原因。原始价格实验留在隔离目录。
- 接入新来源前验证字段、单位、价格口径和日期；历史来源名称如实保留。停用功能在页面明确说明。
- 为行为变化补充有意义的离线验证，尤其覆盖失败保留旧数据的路径。不要让测试真实调用 AI、发邮件或下载行情。
- 仅提交源码、公开配置、脱敏示例和文档。不要提交 `.env`、个人关注列表、持仓、行情、模型、日志、日报、虚拟环境、第三方工作副本或设计临时素材。
- 提交前检查 `git status --short` 与 `git diff --cached`；不要用 `git add .` 将其他工作或个人文件一并带入。

## 反馈问题

在 [Issues](https://github.com/baek0418/AStockAI/issues) 中提供操作系统、Python 版本、执行命令、预期行为与脱敏后的错误日志。勿提交密钥、邮箱授权码或真实持仓。PR 描述应说明问题、最终行为与验证结果。
