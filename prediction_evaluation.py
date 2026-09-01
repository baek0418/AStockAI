"""未来 5 日上涨概率研究的 JSON 与 Markdown 评估报告。"""

import json
from datetime import datetime
from pathlib import Path

from prediction_features import FEATURE_COLUMNS, LABEL_COLUMN, RESEARCH_POOL_EXCLUSION_REASON


def serialize_window(window):
    """将 numpy/pandas 日期等窗口字段转换为 JSON 可保存的形式。"""
    serialized = dict(window)
    calibration = serialized.get("校准")
    if isinstance(calibration, dict):
        model_dates = calibration.get("model_train_dates", [])
        calibration_dates = calibration.get("calibration_dates", [])
        serialized["校准"] = {
            **calibration,
            "模型训练日期范围": [str(model_dates[0]), str(model_dates[-1])] if model_dates else [],
            "模型训练日期数": len(model_dates),
            "校准日期范围": [str(calibration_dates[0]), str(calibration_dates[-1])] if calibration_dates else [],
            "校准日期数": len(calibration_dates),
        }
        serialized["校准"].pop("model_train_dates", None)
        serialized["校准"].pop("calibration_dates", None)
    return serialized


def create_evaluation_data(
    dataset,
    evaluation,
    skipped_files,
    sklearn_version,
    *,
    model_version="v5.0-hgb-5d-up-baseline",
    feature_columns=FEATURE_COLUMNS,
    label_column=LABEL_COLUMN,
    label_definition="future_5d_return = close[t+5] / close[t] - 1；target_up_5d = 1 当 future_5d_return > 0，否则为 0。",
    data_scope_extra=None,
    baseline_label="永远预测上涨率基线",
    baseline_probability_display_key="预测上涨概率",
    outcome_description="上涨与下跌",
    actual_positive_rate_key="实际上涨率",
    report_title="AStockAI 未来 5 日上涨概率研究基线评估",
):
    """组合训练样本摘要、滚动结果、校准分桶与风险提示。"""
    successful_windows = [window for window in evaluation.get("windows", []) if window.get("status") == "success"]
    risks = []
    if not evaluation.get("ready"):
        risks.append(evaluation.get("message", "样本外验证未完成。"))
    if any(not window.get("校准", {}).get("calibrated", False) for window in successful_windows):
        risks.append("至少一个滚动窗口未完成时间 sigmoid 校准，概率可比性有限。")
    research_pool_excluded = [
        item for item in skipped_files if item.get("reason") == RESEARCH_POOL_EXCLUSION_REASON
    ]
    data_problem_skipped = [item for item in skipped_files if item not in research_pool_excluded]
    if data_problem_skipped:
        risks.append(f"有 {len(data_problem_skipped)} 个正式历史 CSV 因数据问题被跳过；应核对其是否影响研究池覆盖。")
    if research_pool_excluded:
        risks.append(f"有 {len(research_pool_excluded)} 个根目录历史 CSV 被研究股票池规则明确排除。")
    aggregate_metrics = evaluation.get("aggregate_metrics")
    baseline_metrics = evaluation.get("aggregate_baseline_metrics")
    if aggregate_metrics and baseline_metrics:
        if aggregate_metrics["brier_score"] >= baseline_metrics["brier_score"]:
            risks.append(f"模型 Brier Score 未优于{baseline_label}，概率质量尚不稳定。")
        if aggregate_metrics.get("roc_auc") is not None and aggregate_metrics["roc_auc"] < 0.55:
            risks.append(f"样本外 ROC-AUC 接近随机水平，模型对{outcome_description}的区分能力有限。")
    risks.extend(
        [
            "样本外结果仅反映历史区间，不代表未来表现。",
            "模型只使用历史日线特征，未包含新闻、公告、资金流或基本面信息。",
            "本研究不构成投资建议，不应用于自动交易。",
        ]
    )
    return {
        "模型版本": model_version,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scikit_learn版本": sklearn_version,
        "特征列表": feature_columns,
        "标签定义": label_definition,
        "数据范围": {
            "训练日期范围": [str(dataset["日期"].min().date()), str(dataset["日期"].max().date())],
            "股票数量": int(dataset["股票名称"].nunique()),
            "样本数": int(len(dataset)),
            "正样本比例": round(float(dataset[label_column].mean()), 6),
            "跳过文件": skipped_files,
            "数据问题跳过文件": data_problem_skipped,
            "研究池外文件": research_pool_excluded,
            **(data_scope_extra or {}),
        },
        "滚动样本外验证": [serialize_window(window) for window in evaluation.get("windows", [])],
        "汇总样本外指标": aggregate_metrics,
        "汇总朴素概率基线": baseline_metrics,
        "汇总永远预测上涨率基线": baseline_metrics,
        "朴素概率基线名称": baseline_label,
        "朴素概率基线概率字段": baseline_probability_display_key,
        "正类描述": outcome_description,
        "实际正类比例字段": actual_positive_rate_key,
        "报告标题": report_title,
        "概率校准分桶": evaluation.get("probability_bins", []),
        "验证状态": evaluation.get("message"),
        "风险提示": risks,
    }


def create_evaluation_markdown(evaluation_data):
    """生成面向人工审阅的研究报告，不输出投资建议。"""
    data_scope = evaluation_data["数据范围"]
    lines = [
        f"# {evaluation_data.get('报告标题', 'AStockAI 未来 5 日上涨概率研究基线评估')}",
        "",
        f"生成时间：{evaluation_data['生成时间']}",
        f"模型版本：{evaluation_data['模型版本']}",
        f"scikit-learn：{evaluation_data['scikit_learn版本']}",
        "",
        "## 数据与标签",
        "",
        f"- 训练日期范围：{data_scope['训练日期范围'][0]} 至 {data_scope['训练日期范围'][1]}",
        f"- 股票数量：{data_scope['股票数量']}",
        f"- 样本数：{data_scope['样本数']}",
        f"- 正样本比例：{data_scope['正样本比例']}",
        f"- 标签：{evaluation_data['标签定义']}",
        "",
        "## 滚动样本外验证",
        "",
        evaluation_data["验证状态"],
        "",
    ]
    data_scope_lines = []
    if data_scope.get("数据问题跳过文件"):
        data_scope_lines.append(f"- 因数据问题跳过文件数：{len(data_scope['数据问题跳过文件'])}")
    if data_scope.get("研究池外文件"):
        data_scope_lines.append(f"- 按研究股票池规则排除文件数：{len(data_scope['研究池外文件'])}")
    section_index = lines.index("## 滚动样本外验证")
    if data_scope_lines:
        data_scope_lines.append("")
    lines[section_index:section_index] = data_scope_lines
    for window in evaluation_data["滚动样本外验证"]:
        lines.extend([f"### 窗口 {window['窗口']}", ""])
        if window.get("status") != "success":
            lines.append(f"- 未完成：{window.get('原因', '未知原因')}")
            lines.append("")
            continue
        model = window["模型"]
        baseline = window[evaluation_data["朴素概率基线名称"]]
        lines.extend(
            [
                f"- 训练日期范围：{' 至 '.join(window['训练日期范围'])}",
                f"- 测试日期范围：{' 至 '.join(window['测试日期范围'])}",
                f"- gap：{window['gap 日期数']} 个交易日",
                f"- 校准：{window['校准']['calibration_method']}",
                f"- 模型：Brier {model['brier_score']}，Log Loss {model['log_loss']}，ROC-AUC {model['roc_auc']}，准确率 {model['accuracy']}",
                f"- {evaluation_data['朴素概率基线名称']}：概率 {baseline[evaluation_data['朴素概率基线概率字段']]}，Brier {baseline['brier_score']}，Log Loss {baseline['log_loss']}，ROC-AUC {baseline['roc_auc']}，准确率 {baseline['accuracy']}",
                "",
            ]
        )
    metrics = evaluation_data.get("汇总样本外指标")
    lines.extend(["## 汇总样本外表现", ""])
    if metrics:
        lines.append(f"- Brier Score：{metrics['brier_score']}")
        lines.append(f"- Log Loss：{metrics['log_loss']}")
        lines.append(f"- ROC-AUC：{metrics['roc_auc']}")
        lines.append(f"- 准确率：{metrics['accuracy']}")
    else:
        lines.append("- 数据不足，未形成有效汇总指标。")
    baseline_metrics = evaluation_data.get("汇总朴素概率基线")
    if baseline_metrics:
        lines.extend(
            [
                "",
                f"- 汇总{evaluation_data['朴素概率基线名称']}："
                f"Brier {baseline_metrics['brier_score']}，Log Loss {baseline_metrics['log_loss']}，"
                f"ROC-AUC {baseline_metrics['roc_auc']}，准确率 {baseline_metrics['accuracy']}",
            ]
        )
    actual_rate_key = evaluation_data.get("实际正类比例字段", "实际上涨率")
    lines.extend([
        "", "## 概率校准分桶", "",
        f"| 概率区间 | 样本数 | 平均预测概率 | {actual_rate_key} |",
        "| --- | ---: | ---: | ---: |",
    ])
    for bucket in evaluation_data["概率校准分桶"]:
        lines.append(
            f"| {bucket['区间']} | {bucket['样本数']} | {bucket['平均预测概率']} | {bucket[actual_rate_key]} |"
        )
    lines.extend(["", "## 风险提示", ""])
    lines.extend(f"- {risk}" for risk in evaluation_data["风险提示"])
    return "\n".join(lines) + "\n"


def save_evaluation_report(evaluation_data, output_directory, prefix="prediction_evaluation"):
    """保存同日期 JSON 与 Markdown 评估报告。"""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().strftime("%Y-%m-%d")
    json_file = output_path / f"{prefix}_{report_date}.json"
    markdown_file = output_path / f"{prefix}_{report_date}.md"
    json_file.write_text(json.dumps(evaluation_data, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_file.write_text(create_evaluation_markdown(evaluation_data), encoding="utf-8")
    return json_file, markdown_file
