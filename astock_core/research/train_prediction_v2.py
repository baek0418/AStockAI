"""训练 v5.1 跑赢市场基准的实验模型，绝不覆盖 v5.0 文件。"""

import json
import sys
from datetime import datetime
from pathlib import Path

from astock_core.research.prediction_evaluation import create_evaluation_data, save_evaluation_report
from astock_core.research.prediction_features_v2 import (
    DEFAULT_BENCHMARK,
    FEATURE_COLUMNS_V2,
    LABEL_COLUMN_V2,
    build_feature_dataset_v2,
    get_labeled_dataset_v2,
)
from astock_core.research.prediction_model import (
    MIN_ROLLING_WINDOWS,
    MIN_TRAIN_SAMPLES,
    RANDOM_STATE,
    evaluate_rolling_windows,
    fit_time_calibrated_model,
    get_sklearn_dependencies,
)
from astock_core.research.train_prediction import get_joblib
from astock_core.runtime.process_journal import ProcessJournal


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()
MODEL_VERSION_V2 = "v5.1-hgb-5d-outperform-benchmark"


def create_metadata(dataset, evaluation_data, calibration_info, sklearn_version, research_ready, benchmark_name):
    """保存独立模型的可追溯研究元数据。"""
    model_dates = calibration_info.get("model_train_dates", [])
    calibration_dates = calibration_info.get("calibration_dates", [])
    calibration = {
        **calibration_info,
        "模型训练日期范围": [str(model_dates[0]), str(model_dates[-1])] if model_dates else [],
        "模型训练日期数": len(model_dates),
        "校准日期范围": [str(calibration_dates[0]), str(calibration_dates[-1])] if calibration_dates else [],
        "校准日期数": len(calibration_dates),
    }
    calibration.pop("model_train_dates", None)
    calibration.pop("calibration_dates", None)
    return {
        "模型版本": MODEL_VERSION_V2,
        "随机种子": RANDOM_STATE,
        "scikit_learn版本": sklearn_version,
        "特征列表": FEATURE_COLUMNS_V2,
        "预测目标": "未来 5 个交易日跑赢市场基准的实验概率",
        "市场基准": benchmark_name,
        "训练截止日期": str(dataset["日期"].max().date()),
        "训练日期范围": [str(dataset["日期"].min().date()), str(dataset["日期"].max().date())],
        "训练样本数": int(len(dataset)),
        "训练股票数": int(dataset["股票名称"].nunique()),
        "正样本比例": round(float(dataset[LABEL_COLUMN_V2].mean()), 6),
        "校准": calibration,
        "滚动验证通过": research_ready,
        "最低要求": {
            "最少训练样本": MIN_TRAIN_SAMPLES,
            "最少滚动窗口": MIN_ROLLING_WINDOWS,
            "至少一个完成 sigmoid 时间校准的窗口": True,
            "样本外 ROC-AUC 不接近随机": ">= 0.55",
            "Brier Score 优于朴素基线": True,
            "Log Loss 优于朴素基线": True,
        },
        "评估报告": evaluation_data,
        "风险提示": "仅作量化研究，不构成投资建议；未通过验证或接近随机时不得接入日报、邮件或 Web。",
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def train_prediction_v2(project_directory=PROJECT_DIRECTORY, benchmark_name=DEFAULT_BENCHMARK):
    """运行固定随机种子、扩张窗口和时间 sigmoid 校准的 v5.1 实验。"""
    project_directory = Path(project_directory)
    journal = ProcessJournal("prediction_v51_training", project_directory)
    journal.event("初始化", "info", 模型版本=MODEL_VERSION_V2, 市场基准=benchmark_name)
    try:
        dependencies = get_sklearn_dependencies()
        joblib = get_joblib()
        features, skipped_files = build_feature_dataset_v2(
            project_directory=project_directory, benchmark_name=benchmark_name
        )
        dataset = get_labeled_dataset_v2(features)
    except (RuntimeError, ValueError) as error:
        journal.event("构建特征", "failed", 原因=str(error))
        return {"status": "failed", "message": str(error)}
    journal.event(
        "构建特征",
        "info",
        样本数=len(dataset),
        股票数=int(dataset["股票名称"].nunique()) if not dataset.empty else 0,
        跳过文件数=len(skipped_files),
        特征数=len(FEATURE_COLUMNS_V2),
    )
    if len(dataset) < MIN_TRAIN_SAMPLES or dataset[LABEL_COLUMN_V2].nunique() < 2:
        journal.event("训练准入", "failed", 原因="研究样本不足或标签只有单一类别。")
        return {"status": "failed", "message": "研究样本不足或标签只有单一类别，拒绝训练。"}
    evaluation = evaluate_rolling_windows(
        dataset,
        dependencies,
        feature_columns=FEATURE_COLUMNS_V2,
        label_column=LABEL_COLUMN_V2,
        probability_column="预测跑赢基准概率",
        baseline_probability_column="基准跑赢概率",
        baseline_label="永远预测正类比例基线",
        baseline_probability_display_key="预测正类概率",
        outcome_description="跑赢市场基准",
        actual_positive_rate_key="实际跑赢基准率",
    )
    evaluation_data = create_evaluation_data(
        dataset,
        evaluation,
        skipped_files,
        dependencies["sklearn"].__version__,
        model_version=MODEL_VERSION_V2,
        feature_columns=FEATURE_COLUMNS_V2,
        label_column=LABEL_COLUMN_V2,
        label_definition=(
            "future_excess_return_5d = future_5d_return - future_market_5d_return；"
            "target_outperform_benchmark_5d = 1 当 future_excess_return_5d > 0，否则为 0。"
        ),
        data_scope_extra={"市场基准": benchmark_name},
        baseline_label="永远预测正类比例基线",
        baseline_probability_display_key="预测正类概率",
        outcome_description="跑赢市场基准",
        actual_positive_rate_key="实际跑赢基准率",
        report_title="AStockAI 未来 5 日跑赢市场基准概率研究评估",
    )
    output_directory = project_directory / "output" / "prediction"
    evaluation_json, evaluation_markdown = save_evaluation_report(
        evaluation_data, output_directory, prefix="prediction_outperform_evaluation"
    )
    journal.event(
        "滚动样本外验证",
        "success" if evaluation.get("ready") else "partial",
        完成窗口数=sum(item.get("status") == "success" for item in evaluation.get("windows", [])),
        评估报告=str(evaluation_json),
    )
    try:
        model, calibration_info = fit_time_calibrated_model(
            dataset, dependencies, FEATURE_COLUMNS_V2, LABEL_COLUMN_V2
        )
    except ValueError as error:
        journal.event("最终模型训练", "failed", 原因=str(error))
        return {
            "status": "failed",
            "message": f"最终模型训练失败：{error}",
            "evaluation_json": str(evaluation_json),
            "evaluation_markdown": str(evaluation_markdown),
        }
    successful_windows = [item for item in evaluation.get("windows", []) if item.get("status") == "success"]
    aggregate = evaluation.get("aggregate_metrics") or {}
    baseline = evaluation.get("aggregate_baseline_metrics") or {}
    research_ready = (
        evaluation.get("ready", False)
        and len(successful_windows) >= MIN_ROLLING_WINDOWS
        and any(item["校准"].get("calibrated", False) for item in successful_windows)
        and aggregate.get("roc_auc") is not None
        and aggregate["roc_auc"] >= 0.55
        and aggregate.get("brier_score", float("inf")) < baseline.get("brier_score", float("inf"))
        and aggregate.get("log_loss", float("inf")) < baseline.get("log_loss", float("inf"))
    )
    metadata = create_metadata(
        dataset, evaluation_data, calibration_info, dependencies["sklearn"].__version__, research_ready, benchmark_name
    )
    models_directory = project_directory / "models"
    models_directory.mkdir(exist_ok=True)
    model_file = models_directory / "predict_5d_outperform_benchmark.joblib"
    metadata_file = models_directory / "predict_5d_outperform_benchmark_metadata.json"
    joblib.dump(model, model_file)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "success" if research_ready else "insufficient"
    journal.event(
        "最终模型训练",
        status,
        滚动验证通过=research_ready,
        模型文件=str(model_file),
        元数据文件=str(metadata_file),
    )
    return {
        "status": status,
        "message": "v5.1 训练与验证完成。" if research_ready else "训练完成，但研究结果不足，不展示预测概率。",
        "model_file": str(model_file),
        "metadata_file": str(metadata_file),
        "evaluation_json": str(evaluation_json),
        "evaluation_markdown": str(evaluation_markdown),
    }


def main():
    result = train_prediction_v2()
    print(result["message"])
    for key in ("model_file", "metadata_file", "evaluation_json", "evaluation_markdown"):
        if result.get(key):
            print(f"{key}：{result[key]}")
    return 0 if result["status"] in {"success", "insufficient"} else 1


if __name__ == "__main__":
    sys.exit(main())
