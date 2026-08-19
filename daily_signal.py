"""基于相邻量化快照生成关注股票的日间变化信号。"""

import json
from pathlib import Path

import pandas as pd

from score import calculate_score_dataframe


def load_json(json_file, description):
    """读取 JSON 文件并提供不包含额外数据来源的错误提示。"""
    try:
        return json.loads(Path(json_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取{description}：{error}。") from error


def snapshot_date(snapshot_file):
    """优先读取快照内日期，用文件名作为稳定的排序后备。"""
    try:
        snapshot = load_json(snapshot_file, "量化快照")
    except ValueError:
        return ""
    return str(snapshot.get("快照日期", ""))


def find_latest_quant_snapshots(output_directory):
    """返回最新和上一个交易日的 quant_snapshot 文件，不下载任何行情。"""
    snapshot_files = list(Path(output_directory).glob("quant_snapshot_*.json"))
    if not snapshot_files:
        raise FileNotFoundError("output 文件夹中没有 quant_snapshot_*.json 文件。")

    snapshot_files.sort(key=lambda file: (snapshot_date(file), file.name))
    latest_file = snapshot_files[-1]
    latest_date = snapshot_date(latest_file)
    previous_files = [file for file in snapshot_files[:-1] if snapshot_date(file) != latest_date]
    return latest_file, previous_files[-1] if previous_files else None


def find_latest_watchlist_snapshot(output_directory):
    """查找最新 watchlist_snapshot 文件；缺失时仍可用 watchlist 与量化快照生成。"""
    files = sorted(Path(output_directory).glob("watchlist_snapshot_*.json"))
    return files[-1] if files else None


def get_stock_code(stock):
    """读取快照兼容的股票代码字段。"""
    for key in ("股票代码", "code", "代码"):
        value = stock.get(key) if isinstance(stock, dict) else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_stock_name(stock):
    """读取快照兼容的股票名称字段。"""
    for key in ("股票名称", "name"):
        value = stock.get(key) if isinstance(stock, dict) else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def create_stock_lookup(stocks):
    """同时按代码和名称建立索引，避免因某一字段缺失丢失已有快照记录。"""
    lookup = {}
    for stock in stocks:
        if not isinstance(stock, dict):
            continue
        code = get_stock_code(stock)
        name = get_stock_name(stock)
        if code:
            lookup[("code", code)] = stock
        if name:
            lookup[("name", name)] = stock
    return lookup


def find_stock(stock_lookup, code, name):
    """优先代码匹配，代码缺失时才按名称匹配。"""
    return stock_lookup.get(("code", code)) or stock_lookup.get(("name", name))


def is_number(value):
    """排除布尔值后的数值判断。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_number(value):
    """保持快照原值的可读表示，不进行指标计算。"""
    if not is_number(value):
        return "数据不足"
    return round(value, 2)


def calculate_change(current_value, previous_value):
    """计算两份既有快照的直接差值；任一缺失即明确标记数据不足。"""
    if not is_number(current_value) or not is_number(previous_value):
        return "数据不足"
    return round(current_value - previous_value, 2)


def get_ma_relation(stock):
    """根据快照中的 MA5、MA20 描述已有均线关系。"""
    ma5 = stock.get("MA5") if stock else None
    ma20 = stock.get("MA20") if stock else None
    if not is_number(ma5) or not is_number(ma20):
        return "数据不足"
    if ma5 > ma20:
        return "MA5 高于 MA20"
    if ma5 < ma20:
        return "MA5 低于 MA20"
    return "MA5 等于 MA20"


def describe_relation_change(current_relation, previous_relation):
    """描述相邻快照中均线关系的变化。"""
    if current_relation == "数据不足" or previous_relation == "数据不足":
        return "数据不足"
    if current_relation == previous_relation:
        return f"维持{current_relation}"
    return f"由{previous_relation}变为{current_relation}"


def get_macd_state(value):
    """仅依据已有 MACD 数值区分零轴正负状态。"""
    if not is_number(value):
        return "数据不足"
    if value > 0:
        return "正值"
    if value < 0:
        return "负值"
    return "零轴"


def describe_macd_change(current_value, previous_value):
    """描述 MACD 正负转换及同侧数值扩大/收窄，不重新计算 MACD。"""
    current_state = get_macd_state(current_value)
    previous_state = get_macd_state(previous_value)
    if "数据不足" in {current_state, previous_state}:
        return "数据不足"
    if current_state != previous_state:
        return f"MACD 由{previous_state}转为{current_state}"
    if current_value == previous_value:
        return f"MACD 维持{current_state}"
    if current_state == "正值":
        return "MACD 正值扩大" if current_value > previous_value else "MACD 正值收窄"
    if current_state == "负值":
        return "MACD 负值收窄" if current_value > previous_value else "MACD 负值扩大"
    return "MACD 零轴状态未变"


def get_risk_label(watchlist_stock, advice):
    """优先读取关注快照风险标签，缺失时只根据已有建议标注。"""
    risk = watchlist_stock.get("risk") if watchlist_stock else None
    if isinstance(risk, str) and risk.strip():
        return risk.strip()
    return "风险" if advice == "风险" else "快照未提供"


def get_trend_label(watchlist_stock, ma_relation):
    """优先使用关注快照趋势；缺失时陈述已有均线关系。"""
    trend = watchlist_stock.get("trend") if watchlist_stock else None
    if isinstance(trend, str) and trend.strip():
        return trend.strip()
    return ma_relation if ma_relation != "数据不足" else "快照未提供"


def classify_signal(score, ma_relation, macd_value, advice, risk_label):
    """以展示规则归类，不影响既有评分、建议或量化算法。"""
    if advice == "风险" or risk_label == "风险":
        return "风险"
    if not is_number(score) or ma_relation == "数据不足" or not is_number(macd_value):
        return "观察"
    if score >= 65 and ma_relation == "MA5 高于 MA20" and macd_value > 0:
        return "偏强"
    if score < 50 or ma_relation == "MA5 低于 MA20" or macd_value < 0:
        return "偏弱"
    return "观察"


def create_observation_conditions(ma_relation, macd_value, rsi):
    """生成不含预测承诺、仅针对现有指标状态的观察条件。"""
    conditions = []
    if ma_relation == "MA5 高于 MA20":
        conditions.append("观察 MA5 是否继续高于 MA20。")
    elif ma_relation == "MA5 低于 MA20":
        conditions.append("观察 MA5 是否继续低于 MA20，或是否回到 MA20 上方。")
    else:
        conditions.append("MA5/MA20 数据不足，无法设置均线观察条件。")

    macd_state = get_macd_state(macd_value)
    if macd_state == "正值":
        conditions.append("观察 MACD 是否继续为正。")
    elif macd_state == "负值":
        conditions.append("观察 MACD 是否转正。")
    elif macd_state == "零轴":
        conditions.append("观察 MACD 是否离开零轴并形成明确方向。")
    else:
        conditions.append("MACD 数据不足，无法设置动量观察条件。")

    if is_number(rsi):
        conditions.append("观察 RSI 是否保持在当前区间，避免仅凭单日变化判断。")
    else:
        conditions.append("RSI 数据不足，无法设置 RSI 观察条件。")
    return conditions


def derive_previous_stock_from_history(current_stock, previous_date, data_directory):
    """在前日快照漏掉新纳入股票时，用截至前日的本地日线补齐比较基线。

    只截取 ``previous_date`` 及以前的行，再复用同一评分函数，因此不会读取
    后续行情；本地历史不足或日期不一致时继续返回 ``None``，不伪造变化项。
    """
    if not current_stock or not previous_date or not data_directory:
        return None
    file_name = current_stock.get("数据文件")
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    file_path = Path(data_directory) / file_name
    try:
        history = pd.read_csv(file_path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return None
    required_columns = {"日期", "收盘", "成交量"}
    if not required_columns.issubset(history.columns):
        return None
    history = history.copy()
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    history["收盘"] = pd.to_numeric(history["收盘"], errors="coerce")
    history["成交量"] = pd.to_numeric(history["成交量"], errors="coerce")
    history = history.dropna(subset=["日期", "收盘", "成交量"]).sort_values("日期")
    history = history[history["日期"] <= pd.Timestamp(previous_date)].copy()
    if len(history) < 26 or history.empty:
        return None
    if history.iloc[-1]["日期"].date().isoformat() != str(previous_date):
        return None
    try:
        result = calculate_score_dataframe(history)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return {
        "综合评分": result["评分"],
        "RSI": result["RSI"],
        "MA5": result["MA5"],
        "MA20": result["MA20"],
        "MACD": result["MACD"],
    }


def create_signal_stock(
    watchlist_config,
    current_stock,
    previous_stock,
    watchlist_stock,
    previous_source="snapshot",
):
    """把一只关注股的当前快照与前一交易日快照整理为可展示的事实。"""
    code = str(watchlist_config.get("code", "")).strip()
    name = str(watchlist_config.get("name", "")).strip()
    if not current_stock:
        return {
            "股票代码": code,
            "股票名称": name,
            "数据状态": "当前 quant_snapshot 中无该股票，现有数据不足，无法判断。",
            "当前指标": {},
            "今日变化": {
                "Score变化": "数据不足",
                "RSI变化": "数据不足",
                "MA5/MA20关系变化": "数据不足",
                "MACD状态变化": "数据不足",
            },
            "信号分类": "观察",
            "观察重点": ["当前量化快照缺少该股票，无法补造历史数据或观察条件。"],
        }

    score = current_stock.get("综合评分")
    rsi = current_stock.get("RSI")
    ma5 = current_stock.get("MA5")
    ma20 = current_stock.get("MA20")
    macd = current_stock.get("MACD")
    advice = current_stock.get("建议", "快照未提供")
    ma_relation = get_ma_relation(current_stock)
    previous_relation = get_ma_relation(previous_stock)
    risk_label = get_risk_label(watchlist_stock, advice)
    trend = get_trend_label(watchlist_stock, ma_relation)
    has_previous = previous_stock is not None
    if previous_source == "local_history":
        data_status = "前一交易日指标由本地日线回溯生成。"
    elif has_previous:
        data_status = "前一交易日快照可用。"
    else:
        data_status = "缺少前一交易日快照，变化项数据不足。"

    return {
        "股票代码": code,
        "股票名称": name,
        "数据状态": data_status,
        "当前指标": {
            "Score": format_number(score),
            "RSI": format_number(rsi),
            "MA5": format_number(ma5),
            "MA20": format_number(ma20),
            "MACD": format_number(macd),
            "趋势": trend,
            "建议": advice,
            "风险标签": risk_label,
            "MA5/MA20关系": ma_relation,
            "MACD状态": get_macd_state(macd),
        },
        "今日变化": {
            "Score变化": calculate_change(score, previous_stock.get("综合评分") if previous_stock else None),
            "RSI变化": calculate_change(rsi, previous_stock.get("RSI") if previous_stock else None),
            "MA5/MA20关系变化": describe_relation_change(ma_relation, previous_relation),
            "MACD状态变化": describe_macd_change(macd, previous_stock.get("MACD") if previous_stock else None),
        },
        "信号分类": classify_signal(score, ma_relation, macd, advice, risk_label),
        "观察重点": create_observation_conditions(ma_relation, macd, rsi),
    }


def build_daily_signal(
    current_snapshot,
    previous_snapshot,
    watchlist_snapshot,
    watchlist_data,
    data_directory=None,
):
    """只使用量化、关注快照和 watchlist 配置构造日间信号 JSON。"""
    current_stocks = current_snapshot.get("股票排行榜", [])
    previous_stocks = previous_snapshot.get("股票排行榜", []) if previous_snapshot else []
    watchlist_stocks = watchlist_snapshot.get("stocks", []) if watchlist_snapshot else []
    if not isinstance(current_stocks, list):
        raise ValueError("最新 quant_snapshot 缺少股票排行榜列表。")
    if not isinstance(previous_stocks, list) or not isinstance(watchlist_stocks, list):
        raise ValueError("快照中的股票列表格式错误。")

    current_lookup = create_stock_lookup(current_stocks)
    previous_lookup = create_stock_lookup(previous_stocks)
    watchlist_lookup = create_stock_lookup(watchlist_stocks)
    watchlist_configs = watchlist_data.get("stocks", []) if isinstance(watchlist_data, dict) else []
    if not isinstance(watchlist_configs, list):
        raise ValueError("watchlist.json 中的 stocks 必须是列表。")

    stocks = []
    for config in watchlist_configs:
        if not isinstance(config, dict) or config.get("enable", True) is False:
            continue
        code = str(config.get("code", "")).strip()
        name = str(config.get("name", "")).strip()
        if not code or not name:
            continue
        current_stock = find_stock(current_lookup, code, name)
        previous_stock = find_stock(previous_lookup, code, name)
        previous_source = "snapshot" if previous_stock else None
        if previous_stock is None and previous_snapshot:
            previous_stock = derive_previous_stock_from_history(
                current_stock,
                previous_snapshot.get("快照日期"),
                data_directory,
            )
            if previous_stock is not None:
                previous_source = "local_history"
        stocks.append(
            create_signal_stock(
                config,
                current_stock,
                previous_stock,
                find_stock(watchlist_lookup, code, name),
                previous_source,
            )
        )

    return {
        "快照日期": current_snapshot.get("快照日期", "未知日期"),
        "前一交易日快照日期": previous_snapshot.get("快照日期") if previous_snapshot else None,
        "前一交易日数据可用": previous_snapshot is not None,
        "stocks": stocks,
    }


def save_daily_signal(signal_data, output_directory):
    """保存每日信号 JSON，不改写量化或关注快照。"""
    report_date = signal_data.get("快照日期", "未知日期")
    output_path = Path(output_directory) / f"daily_signal_{report_date}.json"
    output_path.write_text(json.dumps(signal_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def run_daily_signal(output_directory=None, watchlist_file=None, data_directory=None):
    """生成最新量化快照对应的 daily_signal 文件。"""
    project_directory = Path(__file__).parent
    output_directory = Path(output_directory or project_directory / "output")
    watchlist_file = Path(watchlist_file or project_directory / "watchlist.json")
    data_directory = Path(data_directory or project_directory / "data")
    current_file, previous_file = find_latest_quant_snapshots(output_directory)
    watchlist_snapshot_file = find_latest_watchlist_snapshot(output_directory)
    current_snapshot = load_json(current_file, "最新 quant_snapshot")
    previous_snapshot = load_json(previous_file, "前一交易日 quant_snapshot") if previous_file else None
    watchlist_snapshot = (
        load_json(watchlist_snapshot_file, "最新 watchlist_snapshot")
        if watchlist_snapshot_file
        else None
    )
    watchlist_data = load_json(watchlist_file, "watchlist.json")
    signal_data = build_daily_signal(
        current_snapshot,
        previous_snapshot,
        watchlist_snapshot,
        watchlist_data,
        data_directory,
    )
    signal_data["当前量化快照文件"] = current_file.name
    signal_data["前一交易日量化快照文件"] = previous_file.name if previous_file else None
    signal_data["关注股票快照文件"] = watchlist_snapshot_file.name if watchlist_snapshot_file else None
    return save_daily_signal(signal_data, output_directory)


if __name__ == "__main__":
    print(run_daily_signal())
