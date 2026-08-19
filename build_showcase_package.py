"""构建可外发的 AStockAI 脱敏展示包。

仅打包白名单中的研究代码、合成样例和演示界面；绝不读取或打包真实数据、
模型、输出、日志、关注列表或任何环境变量文件。
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
RELEASE_DATE = datetime.now().strftime("%Y%m%d")
PACKAGE_NAME = f"AStockAI_Showcase_{RELEASE_DATE}"
SOURCE_FILES = [
    "market_data_adapter.py",
    "market_data_provenance.py",
    "qlib_local_export.py",
    "run_qlib_alpha_baseline.py",
    "prediction_features.py",
    "prediction_features_v2.py",
    "prediction_model.py",
    "prediction_data_audit.py",
]


SHOWCASE_README = """# AStockAI 脱敏展示版

这是用于技术交流与产品演示的脱敏版本，不包含真实用户数据、真实股票池、模型权重、日报、日志、账户信息或 API 密钥。

## 快速开始

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-demo.txt
.venv/bin/streamlit run demo_app.py
```

浏览器打开的页面仅展示匿名化标的、合成行情曲线和研究流程摘要。

## 展示的技术能力

- 本地 A 股 OHLCV 标准化与数据质量审计；
- 来源/复权一致性核验：只有已验证前复权数据进入研究集；
- Qlib 二进制数据导出与读取校验；
- Alpha158 因子框架、时间切分的样本外评估与 IC / Rank IC 指标；
- 预测、研究与产品展示分层，未验证结果不会进入推荐界面。

## 明确未包含的内容

- 真实股票名称、代码、历史行情、指数数据及按需查询缓存；
- 模型文件、特征输出、回测或日报结果；
- 环境变量文件、邮件或 AI 服务配置、本机路径与定时任务；
- 实盘交易、自动下单及任何投资建议。

## 数据与模型边界

此展示包中的样例行情均由固定随机种子生成，仅用于说明数据流和 UI，不能用于任何投资判断。展示指标为小规模、匿名研究集上的流程示例，不构成收益证明。

完整 Qlib 实验需要另行安装 Qlib；其源代码与数据均未打包。请遵循上游项目的许可证与数据源服务条款。
"""


DEMO_APP = '''"""AStockAI 脱敏展示界面：只加载包内合成数据。"""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
prices = pd.read_csv(ROOT / "sample_data" / "anonymous_price_history.csv")
metrics = pd.read_json(ROOT / "showcase_metrics.json", typ="series")

st.set_page_config(page_title="AStockAI Showcase", layout="wide")
st.title("AStockAI · 脱敏技术展示")
st.caption("仅使用合成数据；不展示真实股票、客户信息、模型权重或交易建议。")

left, middle, right = st.columns(3)
left.metric("匿名研究标的", f"{metrics['anonymous_instruments']} 个")
middle.metric("样本外 IC（示例）", f"{metrics['out_of_sample_ic']:.2f}")
right.metric("样本外 Rank IC（示例）", f"{metrics['out_of_sample_rank_ic']:.2f}")

st.subheader("研究流程")
st.code("OHLCV 审计 → 复权来源核验 → Qlib 数据集 → Alpha 因子 → 时间切分验证 → 只读展示", language=None)

instrument = st.selectbox("匿名标的", sorted(prices["instrument"].unique()))
history = prices[prices["instrument"] == instrument].copy().set_index("date")
st.line_chart(history[["close"]])

st.subheader("安全与边界")
st.write("真实行情、模型权重、日志、账户配置、环境变量与自动交易功能均未随展示包提供。")
st.info("该页面仅用于展示研究工程流程，不构成投资建议或收益承诺。")
'''


def write_synthetic_sample(sample_directory):
    """生成匿名、不可反推真实行情的演示数据。"""
    rng = np.random.default_rng(20260818)
    dates = pd.bdate_range("2025-01-02", periods=180)
    records = []
    for number in range(1, 6):
        price = 80 + number * 12
        for date in dates:
            price *= 1 + rng.normal(0.0003, 0.014)
            records.append({"date": date.strftime("%Y-%m-%d"), "instrument": f"DEMO_{number:03d}", "close": round(price, 4)})
    pd.DataFrame(records).to_csv(sample_directory / "anonymous_price_history.csv", index=False)


def build_package(project_directory=PROJECT_DIRECTORY):
    """创建目录与 zip；遇到同名旧包拒绝覆盖，避免误删用户已有文件。"""
    project_directory = Path(project_directory)
    dist_directory = project_directory / "dist"
    package_directory = dist_directory / PACKAGE_NAME
    zip_file = dist_directory / f"{PACKAGE_NAME}.zip"
    if package_directory.exists() or zip_file.exists():
        raise FileExistsError(f"展示包已存在：{package_directory}；为安全起见未覆盖。")
    dist_directory.mkdir(exist_ok=True)
    package_directory.mkdir()

    for relative_path in SOURCE_FILES:
        source = project_directory / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"白名单源文件缺失：{relative_path}")
        shutil.copy2(source, package_directory / relative_path)
    (package_directory / "SHOWCASE_README.md").write_text(SHOWCASE_README, encoding="utf-8")
    (package_directory / "demo_app.py").write_text(DEMO_APP, encoding="utf-8")
    (package_directory / "requirements-demo.txt").write_text("streamlit\npandas\nnumpy\n", encoding="utf-8")
    (package_directory / "THIRD_PARTY_NOTICES.md").write_text(
        "Qlib 与 easy-tdx 未被打包；如另行使用，请遵循其各自 MIT 许可证与上游说明。\n",
        encoding="utf-8",
    )
    sample_directory = package_directory / "sample_data"
    sample_directory.mkdir()
    write_synthetic_sample(sample_directory)
    (package_directory / "showcase_metrics.json").write_text(
        json.dumps(
            {
                "anonymous_instruments": 20,
                "out_of_sample_ic": 0.08,
                "out_of_sample_rank_ic": 0.06,
                "note": "Rounded research-flow illustration on an anonymized limited universe; not performance evidence.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(package_directory.rglob("*")):
            if file_path.is_file():
                # 不继承源文件的时间、扩展属性或 Finder 元数据，避免暴露开发环境线索。
                entry = zipfile.ZipInfo(
                    file_path.relative_to(dist_directory).as_posix(),
                    date_time=(datetime.now().year, datetime.now().month, datetime.now().day, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, file_path.read_bytes())
    return package_directory, zip_file


if __name__ == "__main__":
    directory, archive = build_package()
    print(f"展示目录：{directory}")
    print(f"下载包：{archive}")
