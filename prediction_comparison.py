"""如实比较 v5.0 上涨基线与 v5.1 跑赢基准实验，不训练也不下载数据。"""

import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).parent.resolve()


def latest_report(output_directory, prefix):
    """读取指定报告前缀中按文件名最新的一份；损坏文件会明确记录。"""
    files = sorted(Path(output_directory).glob(f"{prefix}_*.json"))
    if not files:
        return None, f"未找到 {prefix} 报告。"
    report_file = files[-1]
    try:
        return json.loads(report_file.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"报告读取失败：{report_file.name}：{error}。"


def metric_summary(report):
    if not report:
        return None
    return {
        "模型版本": report.get("模型版本"),
        "标签定义": report.get("标签定义"),
        "数据范围": report.get("数据范围"),
        "样本外指标": report.get("汇总样本外指标"),
        "朴素概率基线": report.get("汇总朴素概率基线") or report.get("汇总永远预测上涨率基线"),
        "风险提示": report.get("风险提示", []),
    }


def assess_v2(v2_report):
    """不挑选窗口；仅使用完整滚动样本外汇总与同期朴素基线判断。"""
    if not v2_report:
        return {
            "状态": "未完成",
            "结论": "尚无 v5.1 样本外报告，无法与 v5.0 或朴素概率基线进行真实比较。",
            "可接入日报或 Web": False,
        }
    model = v2_report.get("汇总样本外指标")
    baseline = v2_report.get("汇总朴素概率基线") or v2_report.get("汇总永远预测上涨率基线")
    if not model or not baseline or model.get("roc_auc") is None:
        return {
            "状态": "数据不足",
            "结论": "v5.1 未形成完整滚动样本外指标，不能展示或接入概率。",
            "可接入日报或 Web": False,
        }
    improves_baseline = (
        model["brier_score"] < baseline["brier_score"]
        and model["log_loss"] < baseline["log_loss"]
        and model["roc_auc"] > baseline.get("roc_auc", 0.5)
    )
    near_random = model["roc_auc"] < 0.55
    if near_random:
        conclusion = "v5.1 ROC-AUC 仍接近随机，不能进入日报或 Web。"
    elif not improves_baseline:
        conclusion = "v5.1 未同时优于朴素概率基线，不能进入日报或 Web。"
    else:
        conclusion = "v5.1 相对朴素基线有改善，但仍仅限研究；须经独立审阅后才可讨论展示。"
    return {
        "状态": "已评估",
        "是否优于朴素概率基线": improves_baseline,
        "是否接近随机": near_random,
        "结论": conclusion,
        # v5.1 是不同标签，仍保持研究隔离，绝不自动批准接入。
        "可接入日报或 Web": False,
    }


def create_markdown(comparison):
    v5 = comparison["v5.0 上涨概率基线"]
    v2 = comparison["v5.1 跑赢基准实验"]
    assessment = comparison["结论"]
    lines = [
        "# AStockAI v5.0 与 v5.1 预测研究对比",
        "",
        f"生成时间：{comparison['生成时间']}",
        "",
        "## 可比性说明",
        "",
        "- v5.0 目标为未来 5 日上涨；v5.1 目标为未来 5 日跑赢市场基准，标签不同，不能把两个 ROC-AUC 当作同一任务的绝对排名。",
        "- 两者均应使用固定随机种子、按日期扩张窗口、5 个交易日 gap 与时间 sigmoid 校准；报告不删除或挑选差窗口。",
        "",
        "## v5.0 上涨概率基线",
        "",
    ]
    if v5:
        metrics = v5.get("样本外指标") or {}
        baseline = v5.get("朴素概率基线") or {}
        lines.extend([
            f"- 模型版本：{v5.get('模型版本')}",
            f"- 样本外：Brier {metrics.get('brier_score')}，Log Loss {metrics.get('log_loss')}，ROC-AUC {metrics.get('roc_auc')}，准确率 {metrics.get('accuracy')}",
            f"- 朴素基线：Brier {baseline.get('brier_score')}，Log Loss {baseline.get('log_loss')}，ROC-AUC {baseline.get('roc_auc')}，准确率 {baseline.get('accuracy')}",
        ])
    else:
        lines.append(f"- {comparison['v5.0 报告状态']}")
    lines.extend(["", "## v5.1 跑赢基准实验", ""])
    if v2:
        metrics = v2.get("样本外指标") or {}
        baseline = v2.get("朴素概率基线") or {}
        scope = v2.get("数据范围") or {}
        lines.extend([
            f"- 模型版本：{v2.get('模型版本')}",
            f"- 市场基准：{scope.get('市场基准', '未记录')}",
            f"- 样本外：Brier {metrics.get('brier_score')}，Log Loss {metrics.get('log_loss')}，ROC-AUC {metrics.get('roc_auc')}，准确率 {metrics.get('accuracy')}",
            f"- 朴素基线：Brier {baseline.get('brier_score')}，Log Loss {baseline.get('log_loss')}，ROC-AUC {baseline.get('roc_auc')}，准确率 {baseline.get('accuracy')}",
        ])
    else:
        lines.append(f"- {comparison['v5.1 报告状态']}")
    lines.extend([
        "",
        "## 结论",
        "",
        f"- v5.1 是否优于朴素概率基线：{assessment.get('是否优于朴素概率基线', '尚不可判断')}",
        f"- v5.1 是否仍接近随机：{assessment.get('是否接近随机', '尚不可判断')}",
        f"- 是否具备进入日报或 Web 的资格：{assessment['可接入日报或 Web']}",
        f"- {assessment['结论']}",
        "",
        "本报告仅作研究审阅，不构成投资建议。",
    ])
    return "\n".join(lines) + "\n"


def create_comparison(project_directory=PROJECT_DIRECTORY):
    output_directory = Path(project_directory) / "output" / "prediction"
    v5_report, v5_error = latest_report(output_directory, "prediction_evaluation")
    v2_report, v2_error = latest_report(output_directory, "prediction_outperform_evaluation")
    comparison = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "v5.0 报告状态": v5_error or "已读取。",
        "v5.1 报告状态": v2_error or "已读取。",
        "v5.0 上涨概率基线": metric_summary(v5_report),
        "v5.1 跑赢基准实验": metric_summary(v2_report),
        "结论": assess_v2(v2_report),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    report_date = datetime.now().strftime("%Y-%m-%d")
    json_file = output_directory / f"prediction_comparison_{report_date}.json"
    markdown_file = output_directory / f"prediction_comparison_{report_date}.md"
    json_file.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_file.write_text(create_markdown(comparison), encoding="utf-8")
    return comparison, json_file, markdown_file


def main():
    comparison, json_file, markdown_file = create_comparison()
    print(comparison["结论"]["结论"])
    print(f"JSON：{json_file}")
    print(f"Markdown：{markdown_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
