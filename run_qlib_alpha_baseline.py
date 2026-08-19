"""运行隔离的 Qlib Alpha158 + LightGBM 五日收益研究基线。

此脚本仅评估样本外预测相关性与误差，不生成交易指令、概率或现有模型文件。
必须先存在当日的 Qlib 导出报告，且报告只包含已核验的前复权数据。
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_LABEL = "Ref($close, -5) / $close - 1"
SEGMENTS = {
    "train": ("2024-05-31", "2025-06-30"),
    "valid": ("2025-07-01", "2026-03-31"),
    "test": ("2026-04-01", "2026-08-11"),
}


def prepare_xy(dataset, segment, data_handler, feature_columns=None):
    """从 Qlib 数据集取得已学习处理的特征和标签，并拒绝缺失标签。"""
    data = dataset.prepare(segment, col_set=["feature", "label"], data_key=data_handler.DK_L)
    features = data["feature"]
    if feature_columns is not None:
        features = features.loc[:, feature_columns]
    labels = data["label"].iloc[:, 0]
    valid = labels.notna() & features.notna().all(axis=1)
    if not valid.any():
        raise ValueError(f"{segment} 分段没有完整的 Alpha158 特征和五日标签。")
    return features.loc[valid], labels.loc[valid]


def daily_correlation(predictions, labels, method="pearson"):
    """按交易日计算横截面 IC / Rank IC，少于两只股票的日期不纳入。"""
    frame = pd.DataFrame({"prediction": predictions, "label": labels}).dropna()
    values = []
    for _, group in frame.groupby(level="datetime"):
        if len(group) >= 2 and group["prediction"].nunique() > 1 and group["label"].nunique() > 1:
            values.append(float(group["prediction"].corr(group["label"], method=method)))
    return values


def run_baseline(project_directory=None):
    """使用本地 Qlib 数据运行单次固定时间切分的研究基线。"""
    project = Path(project_directory or Path(__file__).parent)
    report_date = datetime.now().strftime("%Y-%m-%d")
    export_file = project / "output" / "research" / f"qlib_export_{report_date}.json"
    if not export_file.exists():
        raise ValueError("缺少当日 Qlib 导出报告；拒绝运行基线。")
    export = json.loads(export_file.read_text(encoding="utf-8"))
    if len(export["selected_instruments"]) < 10:
        raise ValueError("已核验前复权标的不足 10 只；拒绝运行横截面基线。")

    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
    except ImportError as error:
        raise RuntimeError("请在 Qlib 隔离环境中执行此脚本。") from error
    try:
        import lightgbm
        from lightgbm import LGBMRegressor

        model_name = "Qlib Alpha158 + LightGBMRegressor"
        model = LGBMRegressor(
            objective="regression", n_estimators=300, learning_rate=0.03, num_leaves=31,
            max_depth=-1, colsample_bytree=0.8, subsample=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=1, verbosity=-1,
        )
        model_version = lightgbm.__version__
        uses_lightgbm = True
    except (ImportError, OSError):
        from sklearn.ensemble import HistGradientBoostingRegressor
        import sklearn

        model_name = "Qlib Alpha158 + HistGradientBoostingRegressor（LightGBM 环境替代）"
        model = HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0, random_state=42,
        )
        model_version = sklearn.__version__
        uses_lightgbm = False

    qlib.init(provider_uri=export["qlib_directory"], region="cn", redis_port=-1)
    handler = Alpha158(
        instruments="all",
        start_time="2024-02-06",
        end_time="2026-08-18",
        fit_start_time=SEGMENTS["train"][0],
        fit_end_time=SEGMENTS["train"][1],
        label=[TARGET_LABEL],
    )
    dataset = DatasetH(handler=handler, segments=SEGMENTS)
    train_raw = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    availability = train_raw["feature"].notna().mean()
    # Alpha158 的 VWAP 相关因子需要成交额；当前源数据没有该字段。仅保留在训练段
    # 至少 99% 可计算的原始 Alpha158 因子，避免用任何估算值填充成交额或 VWAP。
    usable_features = availability[availability >= 0.99].index.tolist()
    if len(usable_features) < 20:
        raise ValueError("可由当前 OHLCV 严格计算的 Alpha158 因子不足 20 个。")
    train_x, train_y = prepare_xy(dataset, "train", DataHandlerLP, usable_features)
    valid_x, valid_y = prepare_xy(dataset, "valid", DataHandlerLP, usable_features)
    test_x, test_y = prepare_xy(dataset, "test", DataHandlerLP, usable_features)
    if uses_lightgbm:
        model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], callbacks=[lightgbm.early_stopping(30, verbose=False)])
        best_iteration = int(model.best_iteration_ or model.n_estimators)
    else:
        model.fit(train_x, train_y)
        best_iteration = int(model.max_iter)
    prediction = pd.Series(model.predict(test_x), index=test_x.index, name="prediction")
    test_ic = daily_correlation(prediction, test_y, "pearson")
    test_rank_ic = daily_correlation(prediction, test_y, "spearman")
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "feature_set": "Alpha158-reduced（仅保留 OHLCV 可严格计算的因子）",
        "alpha158_total_features": int(train_raw["feature"].shape[1]),
        "usable_features": int(len(usable_features)),
        "qlib_version": qlib.__version__,
        "model_library_version": model_version,
        "target": "未来 5 个交易日收益率（训练标签经 Qlib 默认横截面标准化）",
        "label_expression": TARGET_LABEL,
        "data_selection": export["selection_rule"],
        "instruments": export["selected_instruments"],
        "excluded_instruments": export["excluded_instruments"],
        "segments": {name: list(value) for name, value in SEGMENTS.items()},
        "samples": {"train": int(len(train_y)), "valid": int(len(valid_y)), "test": int(len(test_y))},
        "best_iteration": best_iteration,
        "test_metrics": {
            "mse_on_standardized_label": float(np.mean((prediction - test_y) ** 2)),
            "mae_on_standardized_label": float(np.mean(np.abs(prediction - test_y))),
            "ic_mean": float(np.mean(test_ic)) if test_ic else None,
            "ic_std": float(np.std(test_ic)) if test_ic else None,
            "rank_ic_mean": float(np.mean(test_rank_ic)) if test_rank_ic else None,
            "rank_ic_std": float(np.std(test_rank_ic)) if test_rank_ic else None,
            "ic_days": len(test_ic),
        },
        "research_status": "baseline_only",
        "limitations": [
            "仅 20 只前复权股票，不代表全 A 股横截面。",
            "源 CSV 缺少成交额，未使用依赖成交额/VWAP 的 Alpha158 因子。",
            "目标是未来五日绝对收益，尚未对沪深300计算超额收益。",
            "MSE/MAE 基于 Qlib 默认横截面标准化后的训练标签，不能解释为实际收益率误差。",
            "未加入涨跌停、停牌、T+1、整手和交易成本，不能作为交易回测或买卖建议。",
            "若模型名称含“环境替代”，则本机缺少 LightGBM 的 libomp 运行库；该结果只验证 Qlib 数据与研究流程。",
        ],
    }
    output_directory = project / "output" / "research"
    prediction_file = output_directory / f"qlib_alpha158_baseline_predictions_{report_date}.csv"
    report_file = output_directory / f"qlib_alpha158_baseline_{report_date}.json"
    pd.DataFrame({"prediction": prediction, "label": test_y}).reset_index().to_csv(prediction_file, index=False)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_file, prediction_file


if __name__ == "__main__":
    result, report_path, prediction_path = run_baseline()
    print(report_path)
    print(prediction_path)
    print(result["test_metrics"])
