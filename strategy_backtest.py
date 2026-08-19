"""AStockAI v2.7：按每日最高 AI 评分选股的策略回测程序。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from score import calculate_score


INITIAL_CAPITAL = 100000.0
BUY_SCORE = 70
HOLDING_DAYS = 5
START_DAY_INDEX = 29
REQUIRED_COLUMNS = {"日期", "收盘", "成交量"}


def is_historical_data_file(file_path):
    """判断 CSV 文件是否包含策略回测所需的历史行情字段。"""
    try:
        columns = set(pd.read_csv(file_path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return False

    return REQUIRED_COLUMNS.issubset(columns)


def load_stock_history(file_path):
    """读取、清理并按日期升序排列一只股票的历史行情。"""
    history_data = pd.read_csv(file_path)
    history_data["日期"] = pd.to_datetime(history_data["日期"], errors="coerce")
    history_data["收盘"] = pd.to_numeric(history_data["收盘"], errors="coerce")
    history_data["成交量"] = pd.to_numeric(history_data["成交量"], errors="coerce")

    history_data = history_data.dropna(subset=["日期", "收盘", "成交量"])
    return history_data.sort_values("日期").reset_index(drop=True)


def load_stock_pool(data_directory):
    """读取 data 文件夹中全部可用历史行情，建立股票池数据。"""
    stock_pool = {}

    for file_path in sorted(data_directory.glob("*.csv")):
        if not is_historical_data_file(file_path):
            continue

        stock_name = file_path.stem.replace("历史", "")
        stock_pool[stock_name] = load_stock_history(file_path)

    return stock_pool


def get_common_trading_dates(stock_pool):
    """取得所有股票都存在行情的共同交易日，保证每天可横向比较评分。"""
    if not stock_pool:
        return []

    date_sets = [set(history_data["日期"]) for history_data in stock_pool.values()]
    common_dates = set.intersection(*date_sets)
    return sorted(common_dates)


def get_history_until_date(history_data, current_date):
    """截取一只股票截至当前交易日可见的历史数据。"""
    return history_data[history_data["日期"] <= current_date]


def get_close_price(history_data, current_date):
    """取得一只股票在指定交易日的收盘价格。"""
    close_prices = history_data.loc[
        history_data["日期"] == current_date,
        "收盘",
    ]
    return float(close_prices.iloc[-1])


def calculate_stock_score(history_data, current_date, temporary_directory):
    """调用既有评分程序，计算一只股票在指定交易日的 AI 评分。"""
    score_file = temporary_directory / "score_input.csv"
    available_history = get_history_until_date(history_data, current_date)
    available_history.to_csv(score_file, index=False, encoding="utf-8-sig")

    score_result = calculate_score(score_file)
    return score_result["评分"]


def find_best_stock(stock_pool, current_date, temporary_directory):
    """计算股票池全部评分，并返回当天评分最高的股票信息。"""
    candidates = []

    for stock_name, history_data in stock_pool.items():
        score = calculate_stock_score(
            history_data,
            current_date,
            temporary_directory,
        )
        candidates.append({"股票": stock_name, "评分": score})

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda candidate: (-candidate["评分"], candidate["股票"]),
    )[0]


def open_position(stock_name, history_data, current_date, sell_index, capital):
    """按指定股票的收盘价使用全部可用资金建立仓位。"""
    buy_price = get_close_price(history_data, current_date)
    shares = capital / buy_price

    return {
        "股票": stock_name,
        "买入日期": current_date,
        "买入价格": buy_price,
        "买入资金": capital,
        "持仓数量": shares,
        "卖出索引": sell_index,
    }


def get_position_value(position, history_data, current_date):
    """按照指定交易日收盘价计算当前持仓的市值。"""
    close_price = get_close_price(history_data, current_date)
    return position["持仓数量"] * close_price


def close_position(position, history_data, current_date):
    """在指定交易日收盘卖出持仓，并生成完整交易记录。"""
    sell_price = get_close_price(history_data, current_date)
    final_capital = position["持仓数量"] * sell_price
    return_rate = (final_capital / position["买入资金"] - 1) * 100

    trade = {
        "股票": position["股票"],
        "买入日期": position["买入日期"].strftime("%Y-%m-%d"),
        "买入价格": position["买入价格"],
        "卖出日期": current_date.strftime("%Y-%m-%d"),
        "卖出价格": sell_price,
        "收益率": return_rate,
    }
    return final_capital, trade


def create_equity_record(current_date, capital):
    """生成一天的资金曲线记录，收益率表示相对初始资金的累计收益率。"""
    return {
        "日期": current_date.strftime("%Y-%m-%d"),
        "资金": capital,
        "收益率": (capital / INITIAL_CAPITAL - 1) * 100,
    }


def calculate_statistics(trades, final_capital):
    """根据真实资金变化和交易记录计算策略回测统计结果。"""
    if not trades:
        return {
            "总交易次数": 0,
            "胜率": 0.0,
            "平均收益率": 0.0,
            "最终资金": final_capital,
            "累计收益率": (final_capital / INITIAL_CAPITAL - 1) * 100,
            "最大盈利": 0.0,
            "最大亏损": 0.0,
        }

    return_rates = [trade["收益率"] for trade in trades]
    winning_count = sum(return_rate > 0 for return_rate in return_rates)

    return {
        "总交易次数": len(trades),
        "胜率": winning_count / len(trades) * 100,
        "平均收益率": sum(return_rates) / len(trades),
        "最终资金": final_capital,
        "累计收益率": (final_capital / INITIAL_CAPITAL - 1) * 100,
        "最大盈利": max(return_rates),
        "最大亏损": min(return_rates),
    }


def save_equity_curve(equity_curve, output_file):
    """将每日资金和累计收益率保存为后续绘图使用的 CSV 文件。"""
    equity_data = pd.DataFrame(equity_curve)
    equity_data.to_csv(output_file, index=False, encoding="utf-8-sig")


def print_report(statistics):
    """按固定格式输出策略回测报告。"""
    print("========================")
    print("AStockAI 策略回测")
    print("========================")
    print(f"总交易次数：{statistics['总交易次数']}")
    print(f"胜率：{statistics['胜率']:.2f}%")
    print(f"平均收益率：{statistics['平均收益率']:.2f}%")
    print(f"最终资金：{statistics['最终资金']:.2f} 元")
    print(f"累计收益率：{statistics['累计收益率']:.2f}%")
    print(f"最大盈利：{statistics['最大盈利']:.2f}%")
    print(f"最大亏损：{statistics['最大亏损']:.2f}%")
    print("========================")


def run_strategy_backtest():
    """执行全股票池选最高分股票、全仓持有五日的策略回测。"""
    project_directory = Path(__file__).parent
    stock_pool = load_stock_pool(project_directory / "data")
    trading_dates = get_common_trading_dates(stock_pool)

    if len(trading_dates) < START_DAY_INDEX + HOLDING_DAYS + 1:
        raise ValueError("历史数据不足，无法从第 30 个交易日开始完成一次回测。")

    current_capital = INITIAL_CAPITAL
    position = None
    trades = []
    equity_curve = []

    with TemporaryDirectory() as temporary_path:
        temporary_directory = Path(temporary_path)

        for day_index, current_date in enumerate(trading_dates):
            if day_index < START_DAY_INDEX:
                continue

            if position is not None:
                held_history = stock_pool[position["股票"]]
                current_capital = get_position_value(
                    position,
                    held_history,
                    current_date,
                )

                if day_index == position["卖出索引"]:
                    current_capital, trade = close_position(
                        position,
                        held_history,
                        current_date,
                    )
                    trades.append(trade)
                    position = None

                equity_curve.append(create_equity_record(current_date, current_capital))
                continue

            if day_index + HOLDING_DAYS >= len(trading_dates):
                equity_curve.append(create_equity_record(current_date, current_capital))
                continue

            best_stock = find_best_stock(
                stock_pool,
                current_date,
                temporary_directory,
            )

            if best_stock["评分"] >= BUY_SCORE:
                selected_history = stock_pool[best_stock["股票"]]
                position = open_position(
                    best_stock["股票"],
                    selected_history,
                    current_date,
                    day_index + HOLDING_DAYS,
                    current_capital,
                )

            equity_curve.append(create_equity_record(current_date, current_capital))

    final_capital = equity_curve[-1]["资金"]
    statistics = calculate_statistics(trades, final_capital)
    save_equity_curve(equity_curve, project_directory / "equity_curve.csv")
    print_report(statistics)
    return trades, equity_curve, statistics


if __name__ == "__main__":
    run_strategy_backtest()
