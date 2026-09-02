"""读取已验证模型，输出单只股票未来 5 日上涨概率研究结果。"""

import json
import sys
from pathlib import Path

from astock_core.research.prediction_features import FEATURE_COLUMNS, build_feature_dataset, get_latest_prediction_rows


PROJECT_DIRECTORY = Path(__file__).parents[2].resolve()


def load_model_artifacts(project_directory):
    """加载模型及元数据；缺失或未验证时不产生概率。"""
    project_directory = Path(project_directory)
    metadata_file = project_directory / "models" / "predict_5d_up_metadata.json"
    model_file = project_directory / "models" / "predict_5d_up.joblib"
    if not metadata_file.is_file() or not model_file.is_file():
        return None, None, "未找到已训练模型；请先运行 train_prediction.py。"
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, None, f"模型元数据读取失败：{error}。"
    if not metadata.get("滚动验证通过"):
        return None, metadata, "研究结果不足，不展示预测概率。"
    try:
        import joblib

        model = joblib.load(model_file)
    except (ImportError, OSError, ValueError) as error:
        return None, metadata, f"模型加载失败：{error}。"
    return model, metadata, None


def predict_probability(stock_code, project_directory=PROJECT_DIRECTORY):
    """用已有正式历史数据预测指定代码，不下载行情。"""
    model, metadata, error_message = load_model_artifacts(project_directory)
    if error_message:
        return {"status": "insufficient", "message": error_message}
    try:
        features, _ = build_feature_dataset(project_directory=project_directory)
    except ValueError as error:
        return {"status": "failed", "message": str(error)}
    latest_rows = get_latest_prediction_rows(features)
    clean_code = str(stock_code).strip().zfill(6)
    matching_rows = latest_rows[latest_rows["股票代码"] == clean_code]
    if matching_rows.empty:
        return {"status": "failed", "message": "未找到该股票的正式历史日线或完整特征。"}
    row = matching_rows.sort_values("日期").iloc[-1]
    probability = float(model.predict_proba(row[FEATURE_COLUMNS].to_frame().T)[:, 1][0])
    return {
        "status": "success",
        "股票": f"{row['股票名称']}（{clean_code}）",
        "数据日期": row["日期"].strftime("%Y-%m-%d"),
        "未来 5 日上涨概率": f"{probability * 100:.0f}%",
        "模型版本": metadata["模型版本"],
        "训练截止日期": metadata["训练截止日期"],
        "模型风险": metadata.get("评估报告", {}).get("风险提示", []),
        "风险提示": "仅作量化研究，不构成投资建议。",
    }


def main():
    """解析预测代码并以中文输出研究结果。"""
    if len(sys.argv) != 2:
        print("用法：.venv/bin/python predict_probability.py 300058")
        return 1
    result = predict_probability(sys.argv[1])
    print(result.get("message", "预测研究完成。"))
    for key in (
        "股票",
        "数据日期",
        "未来 5 日上涨概率",
        "模型版本",
        "训练截止日期",
        "模型风险",
        "风险提示",
    ):
        if key in result:
            if key == "模型风险":
                print("模型风险：")
                for risk in result[key]:
                    print(f"- {risk}")
            else:
                print(f"{key}：{result[key]}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
