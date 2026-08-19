"""v5.1 基准感知预测实验的无泄漏特征与跑赢基准标签。"""

from pathlib import Path

import pandas as pd

from prediction_benchmark_data import BENCHMARKS
from prediction_features import (
    FEATURE_COLUMNS,
    HORIZON_DAYS,
    RETURN_COLUMN,
    build_feature_dataset,
    load_history_csv,
)


BENCHMARK_FEATURE_COLUMNS = [
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility_20d",
    "relative_strength_5d",
    "relative_strength_20d",
    "trend_direction_agreement",
    "relative_volatility_20d",
]
FEATURE_COLUMNS_V2 = [*FEATURE_COLUMNS, *BENCHMARK_FEATURE_COLUMNS]
EXCESS_RETURN_COLUMN = "future_excess_return_5d"
LABEL_COLUMN_V2 = "target_outperform_benchmark_5d"
DEFAULT_BENCHMARK = "沪深300"


def load_benchmark_feature_frame(market_directory, benchmark_name=DEFAULT_BENCHMARK):
    """加载已存在的市场 CSV 并只由当日及过去计算市场特征。"""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"未知市场基准：{benchmark_name}。")
    market_file = Path(market_directory) / BENCHMARKS[benchmark_name]["file_name"]
    if not market_file.is_file():
        raise ValueError(
            f"缺少市场基准数据：{market_file}。请由用户显式运行 prediction_benchmark_data.py 后再训练。"
        )
    history = load_history_csv(market_file)
    if len(history) <= HORIZON_DAYS:
        raise ValueError(f"市场基准 {benchmark_name} 历史日线不足，无法构造未来 5 日标签。")
    benchmark = history[["日期", "收盘"]].copy().sort_values("日期").reset_index(drop=True)
    close = benchmark["收盘"]
    returns = close.pct_change()
    benchmark["market_return_1d"] = close.pct_change(1)
    benchmark["market_return_5d"] = close.pct_change(5)
    benchmark["market_return_20d"] = close.pct_change(20)
    benchmark["market_volatility_20d"] = returns.rolling(20).std()
    benchmark["market_ma5"] = close.rolling(5).mean()
    benchmark["market_ma20"] = close.rolling(20).mean()
    benchmark["market_trend_positive"] = (benchmark["market_ma5"] >= benchmark["market_ma20"]).astype("float")
    benchmark["future_market_5d_return"] = close.shift(-HORIZON_DAYS) / close - 1
    return benchmark


def build_feature_dataset_v2(
    data_directory=None,
    market_directory=None,
    project_directory=None,
    benchmark_name=DEFAULT_BENCHMARK,
):
    """内连接正式股票与基准交易日；不补齐基准缺失日期，也不读取按需目录。"""
    project_directory = Path(project_directory or Path(__file__).parent)
    market_directory = Path(market_directory or project_directory / "data" / "market")
    stock_features, skipped_files = build_feature_dataset(
        data_directory=data_directory, project_directory=project_directory
    )
    benchmark = load_benchmark_feature_frame(market_directory, benchmark_name)
    dataset = stock_features.merge(
        benchmark,
        on="日期",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_市场"),
    )
    if dataset.empty:
        raise ValueError("正式股票历史与市场基准没有可对齐的交易日，拒绝训练。")
    dataset["relative_strength_5d"] = dataset["return_5d"] - dataset["market_return_5d"]
    dataset["relative_strength_20d"] = dataset["return_20d"] - dataset["market_return_20d"]
    stock_trend_positive = dataset["ma5_ma20_deviation"] >= 0
    dataset["trend_direction_agreement"] = (
        stock_trend_positive == dataset["market_trend_positive"].astype(bool)
    ).astype("float")
    market_volatility = dataset["market_volatility_20d"].where(
        dataset["market_volatility_20d"] != 0
    )
    dataset["relative_volatility_20d"] = dataset["volatility_20d"] / market_volatility
    dataset[EXCESS_RETURN_COLUMN] = dataset[RETURN_COLUMN] - dataset["future_market_5d_return"]
    dataset[LABEL_COLUMN_V2] = (dataset[EXCESS_RETURN_COLUMN] > 0).astype("float")
    dataset.loc[
        dataset[RETURN_COLUMN].isna() | dataset["future_market_5d_return"].isna(),
        [EXCESS_RETURN_COLUMN, LABEL_COLUMN_V2],
    ] = float("nan")
    dataset["市场基准"] = benchmark_name
    keep_columns = [
        "股票代码",
        "股票名称",
        "日期",
        "收盘",
        "市场基准",
        *FEATURE_COLUMNS_V2,
        RETURN_COLUMN,
        "future_market_5d_return",
        EXCESS_RETURN_COLUMN,
        LABEL_COLUMN_V2,
    ]
    return dataset[keep_columns].sort_values(["日期", "股票名称"]).reset_index(drop=True), skipped_files


def get_labeled_dataset_v2(feature_dataset):
    """训练仅保留完整特征和有效的未来 5 日跑赢基准标签。"""
    return (
        feature_dataset.dropna(subset=[*FEATURE_COLUMNS_V2, LABEL_COLUMN_V2])
        .copy()
        .sort_values(["日期", "股票名称"])
        .reset_index(drop=True)
    )


def get_latest_prediction_rows_v2(feature_dataset):
    """返回每只股票最新完整特征行；预测阶段不需要未来标签。"""
    complete = feature_dataset.dropna(subset=FEATURE_COLUMNS_V2).copy()
    if complete.empty:
        return complete
    return complete.sort_values("日期").groupby("股票名称", as_index=False).tail(1)
