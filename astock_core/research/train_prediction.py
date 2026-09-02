"""训练并评估未来 5 日上涨概率研究基线。"""

import json
import sys
from datetime import datetime
from pathlib import Path

from astock_core.research.prediction_evaluation import create_evaluation_data, save_evaluation_report
from astock_core.research.prediction_features import FEATURE_COLUMNS, build_feature_dataset, get_labeled_dataset
from astock_core.research.prediction_model import (
    MIN_ROLLING_WINDOWS,
    MIN_TRAIN_SAMPLES,
    MODEL_VERSION,
    RANDOM_STATE,
    evaluate_rolling_windows,
    fit_time_calibrated_model,
    get_sklearn_dependencies,
)
from astock_core.runtime.process_journal import ProcessJournal


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()
MODELS_DIRECTORY = PROJECT_DIRECTORY / "models"
PREDICTION_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "prediction"


def get_joblib():
    """延迟导入 joblib，使缺失依赖时明确拒绝训练。"""
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("缺少 joblib，请先安装 requirements.txt 中的依赖。") from error
    return joblib


def create_metadata(dataset, evaluation_data, calibration_info, sklearn_version, research_ready):
    """记录模型可追溯信息和是否允许展示概率。"""
    model_dates = calibration_info.get("model_train_dates", [])
    calibration_dates = calibration_info.get("calibration_dates", [])
    calibration_metadata = {
        **calibration_info,
        "模型训练日期范围": [str(model_dates[0]), str(model_dates[-1])] if model_dates else [],
        "模型训练日期数": len(model_dates),
        "校准日期范围": [str(calibration_dates[0]), str(calibration_dates[-1])] if calibration_dates else [],
        "校准日期数": len(calibration_dates),
    }
    calibration_metadata.pop("model_train_dates", None)
    calibration_metadata.pop("calibration_dates", None)
    return {
        "模型版本": MODEL_VERSION,
        "随机种子": RANDOM_STATE,
        "scikit_learn版本": sklearn_version,
        "特征列表": FEATURE_COLUMNS,
        "预测目标": "未来 5 个交易日上涨概率",
        "训练截止日期": str(dataset["日期"].max().date()),
        "训练日期范围": [str(dataset["日期"].min().date()), str(dataset["日期"].max().date())],
        "训练样本数": int(len(dataset)),
        "训练股票数": int(dataset["股票名称"].nunique()),
        "正样本比例": round(float(dataset["target_up_5d"].mean()), 6),
        "校准": calibration_metadata,
        "滚动验证通过": research_ready,
        "最低要求": {
            "最少训练样本": MIN_TRAIN_SAMPLES,
            "最少滚动窗口": MIN_ROLLING_WINDOWS,
            "至少一个完成 sigmoid 时间校准的窗口": True,
        },
        "评估报告": evaluation_data,
        "风险提示": "仅作量化研究，不构成投资建议；未通过验证不得展示预测概率。",
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def train_prediction(project_directory=PROJECT_DIRECTORY):
    """构建数据、滚动评估、训练最终模型并安全保存研究元数据。"""
    project_directory = Path(project_directory)
    journal = ProcessJournal("prediction_v50_training", project_directory)
    journal.event("初始化", "info", 模型版本=MODEL_VERSION, 预测目标="未来 5 个交易日上涨概率")
    try:
        dependencies = get_sklearn_dependencies()
        joblib = get_joblib()
    except RuntimeError as error:
        journal.event("加载依赖", "failed", 原因=str(error))
        return {"status": "failed", "message": str(error)}
    try:
        features, skipped_files = build_feature_dataset(project_directory=project_directory)
        dataset = get_labeled_dataset(features)
    except ValueError as error:
        journal.event("构建特征", "failed", 原因=str(error))
        return {"status": "failed", "message": str(error)}
    journal.event(
        "构建特征",
        "info",
        样本数=len(dataset),
        股票数=int(dataset["股票名称"].nunique()) if not dataset.empty else 0,
        跳过文件数=len(skipped_files),
        特征数=len(FEATURE_COLUMNS),
    )
    if len(dataset) < MIN_TRAIN_SAMPLES or dataset["target_up_5d"].nunique() < 2:
        journal.event("训练准入", "failed", 原因="研究样本不足或标签只有单一类别。")
        return {"status": "failed", "message": "研究样本不足或标签只有单一类别，拒绝训练。"}

    evaluation = evaluate_rolling_windows(dataset, dependencies)
    evaluation_data = create_evaluation_data(
        dataset, evaluation, skipped_files, dependencies["sklearn"].__version__
    )
    prediction_output_directory = project_directory / "output" / "prediction"
    evaluation_json, evaluation_markdown = save_evaluation_report(
        evaluation_data, prediction_output_directory
    )
    journal.event(
        "滚动样本外验证",
        "success" if evaluation.get("ready") else "partial",
        完成窗口数=sum(item.get("status") == "success" for item in evaluation.get("windows", [])),
        评估报告=str(evaluation_json),
    )
    try:
        model, calibration_info = fit_time_calibrated_model(dataset, dependencies)
    except ValueError as error:
        journal.event("最终模型训练", "failed", 原因=str(error))
        return {
            "status": "failed",
            "message": f"最终模型训练失败：{error}",
            "evaluation_json": str(evaluation_json),
            "evaluation_markdown": str(evaluation_markdown),
        }

    successful_windows = [
        window for window in evaluation.get("windows", []) if window.get("status") == "success"
    ]
    research_ready = (
        evaluation.get("ready", False)
        and len(successful_windows) >= MIN_ROLLING_WINDOWS
        and any(window["校准"].get("calibrated", False) for window in successful_windows)
    )
    metadata = create_metadata(
        dataset,
        evaluation_data,
        calibration_info,
        dependencies["sklearn"].__version__,
        research_ready,
    )
    models_directory = project_directory / "models"
    models_directory.mkdir(exist_ok=True)
    model_file = models_directory / "predict_5d_up.joblib"
    metadata_file = models_directory / "predict_5d_up_metadata.json"
    joblib.dump(model, model_file)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "success" if research_ready else "insufficient"
    message = (
        "训练与验证完成。"
        if research_ready
        else "训练完成，但未达到滚动验证要求；研究结果不足，不展示预测概率。"
    )
    journal.event(
        "最终模型训练",
        status,
        滚动验证通过=research_ready,
        模型文件=str(model_file),
        元数据文件=str(metadata_file),
    )
    return {
        "status": status,
        "message": message,
        "model_file": str(model_file),
        "metadata_file": str(metadata_file),
        "evaluation_json": str(evaluation_json),
        "evaluation_markdown": str(evaluation_markdown),
    }


def main():
    """输出可供命令行调用的训练结果。"""
    result = train_prediction()
    print(result["message"])
    for key in ("model_file", "metadata_file", "evaluation_json", "evaluation_markdown"):
        if result.get(key):
            print(f"{key}：{result[key]}")
    return 0 if result["status"] in {"success", "insufficient"} else 1


if __name__ == "__main__":
    sys.exit(main())
