"""未来 5 日上涨概率研究的无泄漏特征构造。"""

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "ma5",
    "ma20",
    "ma60",
    "close_ma5_deviation",
    "close_ma20_deviation",
    "close_ma60_deviation",
    "ma5_ma20_deviation",
    "ma20_ma60_deviation",
    "rsi_14",
    "macd",
    "volume_relative_5d",
    "volume_relative_20d",
]
LABEL_COLUMN = "target_up_5d"
RETURN_COLUMN = "future_5d_return"
HORIZON_DAYS = 5
REQUIRED_COLUMNS = {"日期", "收盘", "成交量"}


def load_history_csv(history_file):
    """读取并规范化正式历史 CSV；不读取按需缓存目录。"""
    try:
        history = pd.read_csv(history_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        history = pd.read_csv(history_file)
    missing = REQUIRED_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(f"历史 CSV 缺少字段：{'、'.join(sorted(missing))}。")
    history = history.copy()
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    for column in ("收盘", "成交量"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["日期", "收盘", "成交量"])
    history = history.sort_values("日期").drop_duplicates("日期", keep="last").reset_index(drop=True)
    return history


def calculate_rsi(close):
    """使用 t 日及以前的收盘价计算 RSI，与既有评分口径一致。"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    return 100 - 100 / (1 + gain / loss)


def build_stock_feature_frame(history, stock_name, stock_code=""):
    """按日期升序计算一只股票的 t 时点特征及 shift(-5) 标签。"""
    data = history.copy().sort_values("日期").reset_index(drop=True)
    close = data["收盘"]
    volume = data["成交量"]
    returns = close.pct_change()
    data["return_1d"] = close.pct_change(1)
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)
    data["return_20d"] = close.pct_change(20)
    data["volatility_5d"] = returns.rolling(5).std()
    data["volatility_10d"] = returns.rolling(10).std()
    data["volatility_20d"] = returns.rolling(20).std()
    data["ma5"] = close.rolling(5).mean()
    data["ma20"] = close.rolling(20).mean()
    data["ma60"] = close.rolling(60).mean()
    data["close_ma5_deviation"] = close / data["ma5"] - 1
    data["close_ma20_deviation"] = close / data["ma20"] - 1
    data["close_ma60_deviation"] = close / data["ma60"] - 1
    data["ma5_ma20_deviation"] = data["ma5"] / data["ma20"] - 1
    data["ma20_ma60_deviation"] = data["ma20"] / data["ma60"] - 1
    data["rsi_14"] = calculate_rsi(close)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    data["macd"] = ema12 - ema26
    data["volume_relative_5d"] = volume / volume.rolling(5).mean()
    data["volume_relative_20d"] = volume / volume.rolling(20).mean()
    data[RETURN_COLUMN] = close.shift(-HORIZON_DAYS) / close - 1
    data[LABEL_COLUMN] = (data[RETURN_COLUMN] > 0).astype("float")
    data.loc[data[RETURN_COLUMN].isna(), LABEL_COLUMN] = float("nan")
    data.insert(0, "股票代码", str(stock_code).strip())
    data.insert(1, "股票名称", stock_name)
    return data[["股票代码", "股票名称", "日期", "收盘", *FEATURE_COLUMNS, RETURN_COLUMN, LABEL_COLUMN]]


def create_stock_code_lookup(project_directory):
    """读取已有股票池代码，仅作为标识列，不作为模型特征。"""
    try:
        from stock_universe import get_enabled_stock_universe

        return {stock["name"]: stock["code"] for stock in get_enabled_stock_universe()}
    except (FileNotFoundError, ValueError, ImportError):
        return {}


def build_feature_dataset(data_directory=None, project_directory=None):
    """仅读取 data 根目录的 CSV，绝不递归读取 data/on_demand。"""
    project_directory = Path(project_directory or Path(__file__).parent)
    data_directory = Path(data_directory or project_directory / "data")
    code_lookup = create_stock_code_lookup(project_directory)
    frames = []
    skipped_files = []
    for history_file in sorted(data_directory.glob("*.csv")):
        stock_name = history_file.stem.replace("历史", "")
        try:
            history = load_history_csv(history_file)
            if len(history) <= HORIZON_DAYS:
                raise ValueError("历史日线不足。")
            frames.append(build_stock_feature_frame(history, stock_name, code_lookup.get(stock_name, "")))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            skipped_files.append({"file": history_file.name, "reason": str(error)})
    if not frames:
        raise ValueError("data 根目录中没有可用于预测研究的正式历史 CSV。")
    dataset = pd.concat(frames, ignore_index=True).sort_values(["日期", "股票名称"]).reset_index(drop=True)
    return dataset, skipped_files


def get_labeled_dataset(feature_dataset):
    """移除缺特征和未来 5 日无标签行，训练阶段只使用可验证样本。"""
    required = [*FEATURE_COLUMNS, LABEL_COLUMN]
    return feature_dataset.dropna(subset=required).copy().sort_values(["日期", "股票名称"]).reset_index(drop=True)


def get_latest_prediction_rows(feature_dataset):
    """返回每只股票最新的完整特征行；无需标签，供离线预测使用。"""
    complete = feature_dataset.dropna(subset=FEATURE_COLUMNS).copy()
    if complete.empty:
        return complete
    return complete.sort_values("日期").groupby("股票名称", as_index=False).tail(1)
