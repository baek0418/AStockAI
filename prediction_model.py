"""时间滚动验证与 HistGradientBoosting 概率模型。"""

from dataclasses import dataclass

import numpy as np

from prediction_features import FEATURE_COLUMNS, LABEL_COLUMN


MODEL_VERSION = "v5.0-hgb-5d-up-baseline"
RANDOM_STATE = 20260724
GAP_DAYS = 5
MIN_TRAIN_SAMPLES = 300
MIN_TEST_SAMPLES = 30
MIN_ROLLING_WINDOWS = 3


def get_sklearn_dependencies():
    """延迟导入机器学习依赖，以便缺依赖时安全拒绝训练。"""
    try:
        import sklearn
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
        try:
            from sklearn.frozen import FrozenEstimator
        except ImportError:
            FrozenEstimator = None
    except ImportError as error:
        raise RuntimeError("缺少 scikit-learn，请先安装 requirements.txt 中的依赖。") from error
    return {
        "sklearn": sklearn,
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "accuracy_score": accuracy_score,
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
        "roc_auc_score": roc_auc_score,
        "FrozenEstimator": FrozenEstimator,
    }


@dataclass(frozen=True)
class RollingWindow:
    """一个按全市场日期切分的扩张窗口。"""

    index: int
    train_dates: tuple
    gap_dates: tuple
    test_dates: tuple


def create_rolling_windows(dates, number_of_windows=MIN_ROLLING_WINDOWS, gap_days=GAP_DAYS):
    """按日期生成扩张训练窗口，所有股票同一日期进入同一时段。"""
    unique_dates = tuple(sorted(set(dates)))
    if len(unique_dates) < 80:
        return []
    initial_train_size = max(40, len(unique_dates) // 2)
    available_test_dates = len(unique_dates) - initial_train_size - gap_days
    if available_test_dates < number_of_windows * 10:
        return []
    test_size = available_test_dates // number_of_windows
    windows = []
    for index in range(number_of_windows):
        train_end = initial_train_size + index * test_size
        test_start = train_end + gap_days
        test_end = test_start + test_size
        if test_end > len(unique_dates):
            break
        windows.append(
            RollingWindow(
                index=index + 1,
                train_dates=unique_dates[:train_end],
                gap_dates=unique_dates[train_end:test_start],
                test_dates=unique_dates[test_start:test_end],
            )
        )
    return windows


def split_by_window(dataset, window):
    """用日期集合切分，不允许同日股票落入不同集合。"""
    train = dataset[dataset["日期"].isin(window.train_dates)].copy()
    test = dataset[dataset["日期"].isin(window.test_dates)].copy()
    return train, test


def make_base_model(dependencies):
    """创建固定随机种子的 HistGradientBoostingClassifier 基线。"""
    return dependencies["HistGradientBoostingClassifier"](
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def fit_time_calibrated_model(
    train_dataset,
    dependencies,
    feature_columns=FEATURE_COLUMNS,
    label_column=LABEL_COLUMN,
):
    """将训练尾部作为时间校准集，优先采用 sigmoid 校准且不使用 isotonic。"""
    unique_dates = sorted(train_dataset["日期"].unique())
    calibration_dates_count = max(10, len(unique_dates) // 5)
    calibration_start = len(unique_dates) - calibration_dates_count
    model_train_dates = unique_dates[: max(0, calibration_start - GAP_DAYS)]
    calibration_dates = unique_dates[calibration_start:]
    model_train = train_dataset[train_dataset["日期"].isin(model_train_dates)]
    calibration = train_dataset[train_dataset["日期"].isin(calibration_dates)]
    if len(model_train) < MIN_TRAIN_SAMPLES or len(set(model_train[label_column])) < 2:
        raise ValueError("模型训练样本不足或只有单一类别。")

    base_model = make_base_model(dependencies)
    base_model.fit(model_train[feature_columns], model_train[label_column].astype(int))
    calibration_ready = len(calibration) >= 30 and len(set(calibration[label_column])) == 2
    if not calibration_ready:
        return base_model, {
            "calibrated": False,
            "calibration_method": "未校准（时间校准样本不足）",
            "model_train_dates": model_train_dates,
            "calibration_dates": calibration_dates,
        }

    calibrator_class = dependencies["CalibratedClassifierCV"]
    frozen_estimator = dependencies.get("FrozenEstimator")
    if frozen_estimator is not None:
        calibrated_model = calibrator_class(
            estimator=frozen_estimator(base_model),
            method="sigmoid",
        )
    else:
        try:
            calibrated_model = calibrator_class(estimator=base_model, method="sigmoid", cv="prefit")
        except TypeError:
            calibrated_model = calibrator_class(base_estimator=base_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(calibration[feature_columns], calibration[label_column].astype(int))
    return calibrated_model, {
        "calibrated": True,
        "calibration_method": "sigmoid（按时间尾部校准）",
        "model_train_dates": model_train_dates,
        "calibration_dates": calibration_dates,
    }


def calculate_metrics(y_true, probabilities, dependencies):
    """计算模型与固定上涨率基线共有的概率和分类指标。"""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    metrics = {
        "brier_score": round(float(dependencies["brier_score_loss"](labels, probabilities)), 6),
        "log_loss": round(float(dependencies["log_loss"](labels, probabilities, labels=[0, 1])), 6),
        "accuracy": round(float(dependencies["accuracy_score"](labels, probabilities >= 0.5)), 6),
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = round(float(dependencies["roc_auc_score"](labels, probabilities)), 6)
    else:
        metrics["roc_auc"] = None
    return metrics


def calculate_probability_bins(y_true, probabilities, actual_positive_rate_key="实际上涨率"):
    """输出概率分桶及对应实际正类比例，默认保持 v5.0 的实际上涨率字段。"""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper)
        sample_count = int(mask.sum())
        bins.append(
            {
                "区间": f"{int(lower * 100)}%–{int(min(upper, 1) * 100)}%",
                "样本数": sample_count,
                "平均预测概率": round(float(probabilities[mask].mean()), 6) if sample_count else None,
                actual_positive_rate_key: round(float(labels[mask].mean()), 6) if sample_count else None,
            }
        )
    return bins


def evaluate_rolling_windows(
    dataset,
    dependencies=None,
    feature_columns=FEATURE_COLUMNS,
    label_column=LABEL_COLUMN,
    probability_column="预测上涨概率",
    baseline_probability_column="基线上涨概率",
    baseline_label="永远预测上涨率基线",
    baseline_probability_display_key="预测上涨概率",
    outcome_description="上涨",
    actual_positive_rate_key="实际上涨率",
    out_of_sample_columns=None,
):
    """执行扩张窗口、5 日 gap 的样本外验证并汇总概率校准结果。"""
    dependencies = dependencies or get_sklearn_dependencies()
    windows = create_rolling_windows(dataset["日期"])
    if len(windows) < MIN_ROLLING_WINDOWS:
        return {
            "ready": False,
            "message": "可用日期不足，无法构建至少 3 个带 5 日 gap 的滚动样本外窗口。",
            "windows": [],
            "out_of_sample": [],
        }
    window_results = []
    all_out_of_sample = []
    all_baseline_probabilities = []
    for window in windows:
        train, test = split_by_window(dataset, window)
        if len(train) < MIN_TRAIN_SAMPLES or len(test) < MIN_TEST_SAMPLES:
            window_results.append(
                {
                    "窗口": window.index,
                    "status": "skipped",
                    "原因": "训练或测试样本不足。",
                    "训练样本数": len(train),
                    "测试样本数": len(test),
                }
            )
            continue
        try:
            model, calibration_info = fit_time_calibrated_model(
                train, dependencies, feature_columns, label_column
            )
            probabilities = model.predict_proba(test[feature_columns])[:, 1]
        except (ValueError, RuntimeError) as error:
            window_results.append({"窗口": window.index, "status": "skipped", "原因": str(error)})
            continue
        baseline_probability = float(train[label_column].mean())
        model_metrics = calculate_metrics(test[label_column], probabilities, dependencies)
        baseline_metrics = calculate_metrics(
            test[label_column], np.full(len(test), baseline_probability), dependencies
        )
        window_results.append(
            {
                "窗口": window.index,
                "status": "success",
                "训练日期范围": [str(min(window.train_dates).date()), str(max(window.train_dates).date())],
                "gap 日期数": len(window.gap_dates),
                "测试日期范围": [str(min(window.test_dates).date()), str(max(window.test_dates).date())],
                "训练样本数": len(train),
                "测试样本数": len(test),
                "训练正样本比例": round(float(train[label_column].mean()), 6),
                "校准": calibration_info,
                "模型": model_metrics,
                baseline_label: {baseline_probability_display_key: round(baseline_probability, 6), **baseline_metrics},
            }
        )
        exported_columns = ["日期", "股票名称", label_column]
        for column in out_of_sample_columns or []:
            if column not in test.columns:
                raise ValueError(f"样本外导出字段不存在：{column}。")
            if column not in exported_columns:
                exported_columns.append(column)
        all_out_of_sample.append(test[exported_columns].assign(**{probability_column: probabilities}))
        all_baseline_probabilities.append(
            test[[label_column]].assign(**{baseline_probability_column: baseline_probability})
        )
    if not all_out_of_sample:
        return {
            "ready": False,
            "message": "所有滚动窗口均因样本或校准不足而未完成。",
            "windows": window_results,
            "out_of_sample": [],
        }
    out_of_sample = np.concatenate(
        [frame[[label_column, probability_column]].to_numpy() for frame in all_out_of_sample]
    )
    baseline_out_of_sample = np.concatenate(
        [frame[[label_column, baseline_probability_column]].to_numpy() for frame in all_baseline_probabilities]
    )
    return {
        "ready": True,
        "message": (
            "滚动样本外验证完成。"
            if outcome_description == "上涨"
            else f"{outcome_description}目标的滚动样本外验证完成。"
        ),
        "windows": window_results,
        "out_of_sample": all_out_of_sample,
        "aggregate_metrics": calculate_metrics(out_of_sample[:, 0], out_of_sample[:, 1], dependencies),
        "aggregate_baseline_metrics": calculate_metrics(
            baseline_out_of_sample[:, 0], baseline_out_of_sample[:, 1], dependencies
        ),
        "probability_bins": calculate_probability_bins(
            out_of_sample[:, 0], out_of_sample[:, 1], actual_positive_rate_key
        ),
    }
