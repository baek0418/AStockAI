"""未来 5 日收益区间的独立研究实验；不生成价格预测或交易指令。"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from prediction_features import (
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    RETURN_COLUMN,
    build_feature_dataset,
    get_labeled_dataset,
)
from prediction_model import create_rolling_windows, split_by_window
from process_journal import ProcessJournal


TARGET_COVERAGE = 0.80
LOWER_QUANTILE = (1 - TARGET_COVERAGE) / 2
UPPER_QUANTILE = 1 - LOWER_QUANTILE
MIN_SUCCESSFUL_WINDOWS = 3
MIN_MODEL_TRAIN_SAMPLES = 300
MIN_CALIBRATION_SAMPLES = 60
MAX_COVERAGE_ERROR = 0.08


def _quantile(values, quantile):
    """兼容不同 NumPy 版本的保守分位数实现。"""
    values = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(values, quantile, interpolation="higher"))


def _interval_metrics(actual, lower, upper, point=None):
    """计算区间覆盖、宽度与中心预测误差；输入均为收益率而不是价格。"""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return {"样本数": 0, "覆盖率": None, "平均区间宽度": None, "中心预测MAE": None}
    actual, lower, upper = actual[valid], lower[valid], upper[valid]
    result = {
        "样本数": int(len(actual)),
        "覆盖率": round(float(((actual >= lower) & (actual <= upper)).mean()), 6),
        "平均区间宽度": round(float((upper - lower).mean()), 6),
        "中心预测MAE": None,
    }
    if point is not None:
        point = np.asarray(point, dtype=float)[valid]
        result["中心预测MAE"] = round(float(np.abs(actual - point).mean()), 6)
    return result


def _volatility_scale(frame):
    """把 20 日日波动率换算为 5 日尺度；极低波动时保留最小正值以稳定校准。"""
    daily_volatility = frame["volatility_20d"].to_numpy(dtype=float)
    return np.maximum(daily_volatility * np.sqrt(HORIZON_DAYS), 1e-4)


def _get_dependencies():
    """延迟加载回归器，缺失依赖时由调用方生成明确的不足报告。"""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as error:
        raise RuntimeError("缺少 scikit-learn，无法执行收益区间实验。") from error
    return {"HistGradientBoostingRegressor": HistGradientBoostingRegressor}


def _make_regressor(dependencies):
    return dependencies["HistGradientBoostingRegressor"](
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=20260901,
    )


def _make_quantile_regressor(dependencies, quantile):
    """拟合收益分位数；与均值模型保持相同复杂度，避免借调参制造优势。"""
    return dependencies["HistGradientBoostingRegressor"](
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=20260901,
    )


def _ordered_bounds(lower, upper):
    """分位数模型偶发交叉时保守排序，避免生成反向区间。"""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.minimum(lower, upper), np.maximum(lower, upper)


def _cqr_scores(actual, lower, upper):
    """CQR 非一致性分数：落在预测区间内为非正，校准分位数决定外扩量。"""
    actual = np.asarray(actual, dtype=float)
    lower, upper = _ordered_bounds(lower, upper)
    return np.maximum(lower - actual, actual - upper)


def _split_train_and_calibration(train_frame):
    """训练尾部留作校准，并留出一个收益观察期，避免校准标签污染模型训练。"""
    dates = tuple(sorted(train_frame["日期"].unique()))
    calibration_dates_count = max(10, len(dates) // 5)
    model_end = len(dates) - calibration_dates_count - HORIZON_DAYS
    if model_end <= 0:
        return None, None, {"原因": "训练窗口不足以同时构造模型训练、5 日隔离和校准集。"}
    model_dates = set(dates[:model_end])
    calibration_dates = set(dates[-calibration_dates_count:])
    model_train = train_frame[train_frame["日期"].isin(model_dates)].copy()
    calibration = train_frame[train_frame["日期"].isin(calibration_dates)].copy()
    if len(model_train) < MIN_MODEL_TRAIN_SAMPLES:
        return None, None, {"原因": "模型训练样本不足。"}
    if len(calibration) < MIN_CALIBRATION_SAMPLES:
        return None, None, {"原因": "校准样本不足。"}
    return model_train, calibration, {
        "模型训练日期范围": [str(min(model_dates).date()), str(max(model_dates).date())],
        "校准日期范围": [str(min(calibration_dates).date()), str(max(calibration_dates).date())],
        "内部隔离交易日数": HORIZON_DAYS,
    }


def evaluate_interval_windows(dataset, dependencies=None):
    """对每个日期滚动窗口训练均值模型，以时间尾部残差构造 80% 预测区间。"""
    dependencies = dependencies or _get_dependencies()
    windows = create_rolling_windows(dataset["日期"], gap_days=HORIZON_DAYS)
    results = []
    for window in windows:
        train, test = split_by_window(dataset, window)
        model_train, calibration, split_info = _split_train_and_calibration(train)
        if model_train is None:
            results.append({"窗口": window.index, "状态": "skipped", **split_info})
            continue
        try:
            model = _make_regressor(dependencies)
            model.fit(model_train[FEATURE_COLUMNS], model_train[RETURN_COLUMN])
            calibration_prediction = model.predict(calibration[FEATURE_COLUMNS])
            calibration_actual = calibration[RETURN_COLUMN].to_numpy(dtype=float)
            absolute_residual = np.abs(calibration_actual - calibration_prediction)
            residual_radius = _quantile(absolute_residual, TARGET_COVERAGE)
            normalized_residual_multiplier = _quantile(
                absolute_residual / _volatility_scale(calibration), TARGET_COVERAGE
            )
            test_prediction = model.predict(test[FEATURE_COLUMNS])
            lower_model = _make_quantile_regressor(dependencies, LOWER_QUANTILE)
            upper_model = _make_quantile_regressor(dependencies, UPPER_QUANTILE)
            lower_model.fit(model_train[FEATURE_COLUMNS], model_train[RETURN_COLUMN])
            upper_model.fit(model_train[FEATURE_COLUMNS], model_train[RETURN_COLUMN])
            cqr_calibration_lower, cqr_calibration_upper = _ordered_bounds(
                lower_model.predict(calibration[FEATURE_COLUMNS]),
                upper_model.predict(calibration[FEATURE_COLUMNS]),
            )
            cqr_adjustment = _quantile(
                _cqr_scores(calibration_actual, cqr_calibration_lower, cqr_calibration_upper),
                TARGET_COVERAGE,
            )
            cqr_test_lower, cqr_test_upper = _ordered_bounds(
                lower_model.predict(test[FEATURE_COLUMNS]),
                upper_model.predict(test[FEATURE_COLUMNS]),
            )
        except (ValueError, RuntimeError) as error:
            results.append({"窗口": window.index, "状态": "skipped", "原因": str(error)})
            continue

        actual = test[RETURN_COLUMN].to_numpy(dtype=float)
        global_residual_metrics = _interval_metrics(
            actual, test_prediction - residual_radius, test_prediction + residual_radius, test_prediction
        )
        adaptive_radius = normalized_residual_multiplier * _volatility_scale(test)
        model_metrics = _interval_metrics(
            actual, test_prediction - adaptive_radius, test_prediction + adaptive_radius, test_prediction
        )
        cqr_metrics = _interval_metrics(
            actual,
            cqr_test_lower - cqr_adjustment,
            cqr_test_upper + cqr_adjustment,
            (cqr_test_lower + cqr_test_upper) / 2,
        )
        calibration_returns = calibration[RETURN_COLUMN].to_numpy(dtype=float)
        unconditional_lower = _quantile(calibration_returns, LOWER_QUANTILE)
        unconditional_upper = _quantile(calibration_returns, UPPER_QUANTILE)
        unconditional_metrics = _interval_metrics(
            actual,
            np.full(len(test), unconditional_lower),
            np.full(len(test), unconditional_upper),
            np.zeros(len(test)),
        )
        volatility_radius = 1.281552 * _volatility_scale(test)
        volatility_metrics = _interval_metrics(actual, -volatility_radius, volatility_radius, np.zeros(len(test)))
        results.append(
            {
                "窗口": window.index,
                "状态": "success",
                "训练日期范围": [str(min(window.train_dates).date()), str(max(window.train_dates).date())],
                "隔离日期范围": [str(min(window.gap_dates).date()), str(max(window.gap_dates).date())],
                "测试日期范围": [str(min(window.test_dates).date()), str(max(window.test_dates).date())],
                "全局残差半径": round(residual_radius, 6),
                "自适应残差倍数": round(normalized_residual_multiplier, 6),
                "CQR校准外扩": round(cqr_adjustment, 6),
                "区间模型": model_metrics,
                "CQR区间候选": cqr_metrics,
                "全局残差区间基线": global_residual_metrics,
                "无条件收益率基线": unconditional_metrics,
                "历史波动率基线": volatility_metrics,
                **split_info,
            }
        )
    return results


def _aggregate_metrics(windows, key):
    records = [item[key] for item in windows if item.get("状态") == "success" and key in item]
    total_samples = sum(item["样本数"] for item in records)
    if not total_samples:
        return {"样本数": 0, "覆盖率": None, "平均区间宽度": None, "中心预测MAE": None}
    return {
        "样本数": total_samples,
        "覆盖率": round(sum(item["覆盖率"] * item["样本数"] for item in records) / total_samples, 6),
        "平均区间宽度": round(sum(item["平均区间宽度"] * item["样本数"] for item in records) / total_samples, 6),
        "中心预测MAE": round(
            sum(item["中心预测MAE"] * item["样本数"] for item in records) / total_samples, 6
        ),
    }


def audit_price_basis(data_directory, dataset):
    """核对行情审计记录；只允许已留痕的未复权快照进入区间研究。"""
    provenance_directory = Path(data_directory) / "provenance"
    raw_codes = [str(code).strip() for code in dataset["股票代码"]]
    codes = sorted({code.zfill(6) for code in raw_codes if code})
    missing_code_stock_count = int(
        dataset.loc[[not code for code in raw_codes], "股票名称"].nunique()
    )
    records, missing, non_raw_records = [], [], []
    for code in codes:
        file_path = provenance_directory / f"{code}.json"
        try:
            record = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(code)
            continue
        if not isinstance(record, dict):
            missing.append(code)
            continue
        adjustment = str(record.get("复权方式", "unknown"))
        records.append({"股票代码": code, "数据源": record.get("数据源", "未记录"), "复权方式": adjustment})
        if adjustment.lower() != "raw":
            non_raw_records.append({"股票代码": code, "复权方式": adjustment})
    reasons = []
    if missing_code_stock_count:
        reasons.append(f"{missing_code_stock_count} 只研究股票缺少可追溯的股票代码。")
    if missing:
        reasons.append(f"{len(missing)} 只研究股票缺少可追溯行情来源记录。")
    if non_raw_records:
        adjustment_summary = "、".join(
            f"{item['股票代码']}（{item['复权方式']}）" for item in non_raw_records[:5]
        )
        remainder = "" if len(non_raw_records) <= 5 else f" 等 {len(non_raw_records)} 只"
        reasons.append(
            f"{len(non_raw_records)} 只研究股票不是明确留痕的未复权快照（{adjustment_summary}{remainder}）；"
            "不能证明过去特征不含后续公司行为信息。"
        )
    return {
        "研究股票数": len(codes),
        "已核对来源记录数": len(records),
        "缺失来源记录数": len(missing),
        "缺失股票代码股票数": missing_code_stock_count,
        "非原始价格记录数": len(non_raw_records),
        "严格时点价格依据可用": not reasons,
        "结论": "可用于严格时点区间研究。" if not reasons else "；".join(reasons),
        "来源样本": records[:10],
    }


def assess_admission(price_basis_audit, windows):
    """以覆盖率、宽度、基线和时点数据依据共同决定是否允许展示。"""
    successful = [item for item in windows if item.get("状态") == "success"]
    model = _aggregate_metrics(successful, "区间模型")
    global_residual = _aggregate_metrics(successful, "全局残差区间基线")
    unconditional = _aggregate_metrics(successful, "无条件收益率基线")
    volatility = _aggregate_metrics(successful, "历史波动率基线")
    reasons = []
    if not price_basis_audit.get("严格时点价格依据可用"):
        reasons.append("行情历史的时点可得性未通过审计。")
    if len(successful) < MIN_SUCCESSFUL_WINDOWS:
        reasons.append(f"成功滚动窗口不足 {MIN_SUCCESSFUL_WINDOWS} 个。")
    if model["覆盖率"] is None or abs(model["覆盖率"] - TARGET_COVERAGE) > MAX_COVERAGE_ERROR:
        reasons.append("模型区间覆盖率未落在预设容忍范围内。")
    if model["覆盖率"] is not None and volatility["覆盖率"] is not None:
        if abs(model["覆盖率"] - TARGET_COVERAGE) > abs(volatility["覆盖率"] - TARGET_COVERAGE) + 0.02:
            reasons.append("模型覆盖率不优于历史波动率基线。")
    if model["平均区间宽度"] is not None and volatility["平均区间宽度"] is not None:
        if model["平均区间宽度"] > volatility["平均区间宽度"] * 1.10:
            reasons.append("模型区间明显宽于历史波动率基线。")
    admitted = not reasons
    return {
        "是否准入": admitted,
        "结论": "研究通过最低展示门槛；仍仅可作为风险范围研究结果。" if admitted else "保持研究隔离，不展示价格区间。",
        "拒绝原因": reasons,
        "区间模型": model,
        "全局残差区间基线": global_residual,
        "无条件收益率基线": unconditional,
        "历史波动率基线": volatility,
    }


def assess_cqr_candidate(windows):
    """严格判断 CQR 是否值得取代简单波动范围；不因“勉强合格”自动升级。"""
    successful = [item for item in windows if item.get("状态") == "success"]
    candidate = _aggregate_metrics(successful, "CQR区间候选")
    volatility = _aggregate_metrics(successful, "历史波动率基线")
    reasons = []
    if len(successful) < MIN_SUCCESSFUL_WINDOWS:
        reasons.append(f"成功滚动窗口不足 {MIN_SUCCESSFUL_WINDOWS} 个。")
    if candidate["覆盖率"] is None or volatility["覆盖率"] is None:
        reasons.append("CQR 或历史波动率基线缺少可比较指标。")
    else:
        if abs(candidate["覆盖率"] - TARGET_COVERAGE) > abs(volatility["覆盖率"] - TARGET_COVERAGE):
            reasons.append("CQR 总体覆盖率没有比历史波动率更接近 80% 目标。")
        if candidate["平均区间宽度"] > volatility["平均区间宽度"]:
            reasons.append("CQR 总体区间没有比历史波动率更窄。")
    for window in successful:
        coverage = window["CQR区间候选"]["覆盖率"]
        if coverage is None or abs(coverage - TARGET_COVERAGE) > MAX_COVERAGE_ERROR:
            reasons.append(f"窗口 {window['窗口']} 的 CQR 覆盖率未落在预设容忍范围内。")
    return {
        "是否值得替代波动率基线": not reasons,
        "结论": "CQR 在所有预设比较中胜出，可进入后续人工审阅。" if not reasons else "CQR 未胜出；保留为研究负结果，不替代简单波动范围。",
        "拒绝原因": reasons,
        "CQR区间候选": candidate,
        "历史波动率基线": volatility,
    }


def select_interval_method(price_basis_audit, cqr_assessment, windows):
    """选择可面向用户解释的区间方法；复杂候选未胜出时明确回退至简单基线。"""
    successful = [item for item in windows if item.get("状态") == "success"]
    volatility = _aggregate_metrics(successful, "历史波动率基线")
    adaptive = _aggregate_metrics(successful, "区间模型")
    if not price_basis_audit.get("严格时点价格依据可用"):
        return {
            "方法": None,
            "说明": "价格口径未通过审计，不生成任何收益区间。",
            "指标": None,
        }
    if cqr_assessment.get("是否值得替代波动率基线"):
        return {
            "方法": "CQR 自适应收益区间",
            "说明": "CQR 在预设的覆盖率、宽度与逐窗口稳定性比较中胜出。",
            "指标": cqr_assessment["CQR区间候选"],
        }
    return {
        "方法": "历史波动率风险范围",
        "说明": "机器学习候选未在预设比较中明确胜出；使用更简单、验证更稳定的风险范围。",
        "指标": volatility,
        "未选用机器学习候选": {
            "波动率自适应残差区间": adaptive,
            "CQR": cqr_assessment.get("CQR区间候选"),
        },
    }


def _write_report(report, output_directory, data_label):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    report_date = str(report.get("数据范围", {}).get("结束日期") or datetime.now().strftime("%Y-%m-%d"))[:10]
    report_file = output_directory / f"return_interval_5d_{data_label}_{report_date}.json"
    temporary_file = report_file.with_suffix(".json.tmp")
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_file, report_file)
    return report_file


def run_return_interval_research(project_directory=None, data_directory=None, data_label="qfq"):
    """运行独立区间研究并保存正、负结果；不下载行情，不更新页面或日报。"""
    project_directory = Path(project_directory or Path(__file__).parent)
    data_directory = Path(data_directory or project_directory / "data")
    journal = ProcessJournal("return_interval_research", project_directory)
    try:
        dataset, skipped_files = build_feature_dataset(
            data_directory=data_directory,
            project_directory=project_directory,
            allowed_stock_codes=None,
        )
        dataset = get_labeled_dataset(dataset)
        if dataset.empty:
            raise ValueError("没有同时具备特征和未来 5 日收益标签的本地样本。")
        price_basis = audit_price_basis(data_directory, dataset)
        windows = evaluate_interval_windows(dataset)
        admission = assess_admission(price_basis, windows)
        cqr_assessment = assess_cqr_candidate(windows)
        selected_method = select_interval_method(price_basis, cqr_assessment, windows)
        report = {
            "实验": "未来5日收益区间",
            "状态": "admitted" if admission["是否准入"] else "rejected",
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "假设": "仅使用 t 日及以前的 OHLCV 特征，分位数回归加独立时间校准（CQR）可构造比历史波动率基线更接近目标且更窄的未来 5 日收益区间。",
            "数据可得性审计": price_basis,
            "泄漏控制": {
                "预测期限交易日": HORIZON_DAYS,
                "滚动窗口隔离交易日": HORIZON_DAYS,
                "模型与校准间隔离交易日": HORIZON_DAYS,
                "特征": "仅使用 t 日及以前的 OHLCV 技术特征。",
                "限制": "动态前复权历史的时点可得性未通过时，结果自动拒绝接入。",
            },
            "样本范围": {
                "日线目录": str(data_directory),
                "股票数": int(dataset["股票名称"].nunique()),
                "样本数": int(len(dataset)),
                "开始日期": str(dataset["日期"].min().date()),
                "结束日期": str(dataset["日期"].max().date()),
                "跳过文件": skipped_files,
            },
            "区间定义": {
                "目标覆盖率": TARGET_COVERAGE,
                "收益率分位点": [LOWER_QUANTILE, UPPER_QUANTILE],
                "说明": "区间针对未来 5 日收益率；保留波动率缩放残差模型和 CQR 候选。不得解释为目标价、收益承诺或买卖信号。",
            },
            "滚动样本外窗口": windows,
            "准入评估": admission,
            "CQR候选评估": cqr_assessment,
            "推荐区间方法": selected_method,
            "方法来源": {
                "CQR": "Romano、Patterson、Candès（2019）Conformalized Quantile Regression；分位数回归上下界以独立时间校准集外扩。",
                "实现": "HistGradientBoostingRegressor 的 quantile 损失；训练、校准和测试保持原有 5 日隔离。",
            },
            "交易成本与组合回测": "不适用：本实验评估预测区间校准，不生成选股、调仓或交易指令。",
            "失败模式": [
                "动态前复权日线可能不具备历史时点可得性。",
                "覆盖率达标但区间过宽，可能没有实际信息价值。",
                "横截面、市场状态或公司行为变化可能使历史校准在未来失效。",
            ],
        }
        report_file = _write_report(report, project_directory / "output" / "research", data_label)
        journal.event("完成5日收益区间研究", report["状态"], 报告=str(report_file), 准入=admission["是否准入"])
        return {"status": report["状态"], "message": admission["结论"], "output_file": str(report_file), "report": report}
    except (OSError, ValueError, RuntimeError) as error:
        journal.event("完成5日收益区间研究", "failed", 原因=str(error))
        return {"status": "failed", "message": str(error), "output_file": None}


def main():
    parser = argparse.ArgumentParser(description="AStockAI 未来 5 日收益区间研究实验。")
    parser.add_argument("--project-directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data-directory", type=Path, help="原始价格实验目录；默认使用正式前复权 data/。")
    parser.add_argument("--data-label", default="qfq", help="报告文件的数据口径标识。")
    arguments = parser.parse_args()
    result = run_return_interval_research(
        arguments.project_directory, arguments.data_directory, arguments.data_label
    )
    print(result["message"])
    if result.get("output_file"):
        print(result["output_file"])
    return 0 if result["status"] in {"admitted", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
