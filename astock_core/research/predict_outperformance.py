"""读取 v5.1 已验证模型，输出跑赢市场基准的实验概率；不下载行情。"""

import json
import sys
from pathlib import Path

from astock_core.research.prediction_features_v2 import (
    FEATURE_COLUMNS_V2,
    build_feature_dataset_v2,
    get_latest_prediction_rows_v2,
)


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()


def load_model_artifacts(project_directory):
    project_directory = Path(project_directory)
    metadata_file = project_directory / "models" / "predict_5d_outperform_benchmark_metadata.json"
    model_file = project_directory / "models" / "predict_5d_outperform_benchmark.joblib"
    if not metadata_file.is_file() or not model_file.is_file():
        return None, None, "未找到 v5.1 已训练模型；请先准备市场基准后运行 train_prediction_v2.py。"
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, None, f"模型元数据读取失败：{error}。"
    if not metadata.get("滚动验证通过"):
        return None, metadata, "研究结果不足，不展示预测概率。"
    try:
        import joblib

        return joblib.load(model_file), metadata, None
    except (ImportError, OSError, ValueError) as error:
        return None, metadata, f"模型加载失败：{error}。"


def predict_outperformance(stock_code, project_directory=PROJECT_DIRECTORY):
    """仅使用正式股票与已下载市场 CSV 预测指定股票。"""
    model, metadata, error_message = load_model_artifacts(project_directory)
    if error_message:
        return {"status": "insufficient", "message": error_message}
    try:
        features, _ = build_feature_dataset_v2(
            project_directory=project_directory, benchmark_name=metadata["市场基准"]
        )
    except ValueError as error:
        return {"status": "failed", "message": str(error)}
    latest_rows = get_latest_prediction_rows_v2(features)
    clean_code = str(stock_code).strip().zfill(6)
    matches = latest_rows[latest_rows["股票代码"] == clean_code]
    if matches.empty:
        return {"status": "failed", "message": "未找到该股票的正式历史日线或完整特征。"}
    row = matches.sort_values("日期").iloc[-1]
    probability = float(model.predict_proba(row[FEATURE_COLUMNS_V2].to_frame().T)[:, 1][0])
    metrics = metadata.get("评估报告", {}).get("汇总样本外指标") or {}
    return {
        "status": "success",
        "股票": f"{row['股票名称']}（{clean_code}）",
        "数据日期": row["日期"].strftime("%Y-%m-%d"),
        "未来 5 个交易日跑赢市场基准的实验概率": f"{probability * 100:.0f}%",
        "使用的基准指数": metadata["市场基准"],
        "模型版本": metadata["模型版本"],
        "模型训练截止日期": metadata["训练截止日期"],
        "最新样本外 ROC-AUC": metrics.get("roc_auc"),
        "风险提示": metadata.get("风险提示"),
    }


def main():
    if len(sys.argv) != 2:
        print("用法：.venv/bin/python predict_outperformance.py 300750")
        return 1
    result = predict_outperformance(sys.argv[1])
    print(result.get("message", "跑赢基准概率研究完成。"))
    for key in (
        "股票", "数据日期", "未来 5 个交易日跑赢市场基准的实验概率", "使用的基准指数",
        "模型版本", "模型训练截止日期", "最新样本外 ROC-AUC", "风险提示",
    ):
        if key in result:
            print(f"{key}：{result[key]}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
