"""AStockAI 第一版基础回测程序。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from score import calculate_score


# 从第 30 个交易日开始计算评分，买入后持有 5 个交易日。
START_DAY_INDEX = 29
HOLDING_DAYS = 5
BUY_SCORE = 70
REQUIRED_COLUMNS = {"日期", "收盘", "成交量"}


def is_historical_data_file(file_path):
    """判断 CSV 文件是否为可用于回测的历史行情数据。"""
    try:
        columns = set(pd.read_csv(file_path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return False

    return REQUIRED_COLUMNS.issubset(columns)


def find_historical_data_files(data_directory):
    """找出 data 文件夹中全部符合格式的历史行情 CSV 文件。"""
    return [
        file_path
        for file_path in sorted(data_directory.glob("*.csv"))
        if is_historical_data_file(file_path)
    ]


def load_historical_data(file_path):
    """读取、清理并按日期升序排列单只股票的历史行情数据。"""
    history_data = pd.read_csv(file_path)
    history_data["日期"] = pd.to_datetime(history_data["日期"], errors="coerce")
    history_data["收盘"] = pd.to_numeric(history_data["收盘"], errors="coerce")
    history_data["成交量"] = pd.to_numeric(history_data["成交量"], errors="coerce")

    history_data = history_data.dropna(subset=["日期", "收盘", "成交量"])
    return history_data.sort_values("日期").reset_index(drop=True)


def get_stock_name(file_path):
    """根据历史数据文件名取得股票名称。"""
    return file_path.stem.replace("历史", "")


def calculate_score_for_day(history_data, day_index, temporary_directory):
    """将截至指定交易日的数据交给既有评分程序，并返回当天评分。"""
    score_file = temporary_directory / "score_input.csv"
    daily_history = history_data.iloc[: day_index + 1]
    daily_history.to_csv(score_file, index=False, encoding="utf-8-sig")

    score_result = calculate_score(score_file)
    return score_result["评分"]


def create_trade(stock_name, history_data, buy_index):
    """根据买入日和固定持有期生成一笔完整交易记录。"""
    sell_index = buy_index + HOLDING_DAYS
    buy_price = float(history_data.iloc[buy_index]["收盘"])
    sell_price = float(history_data.iloc[sell_index]["收盘"])
    return_rate = (sell_price - buy_price) / buy_price * 100

    return {
        "股票": stock_name,
        "买入日期": history_data.iloc[buy_index]["日期"].strftime("%Y-%m-%d"),
        "买入价格": buy_price,
        "卖出日期": history_data.iloc[sell_index]["日期"].strftime("%Y-%m-%d"),
        "卖出价格": sell_price,
        "收益率": return_rate,
    }


def backtest_stock(stock_name, history_data, temporary_directory):
    """回测单只股票：评分达标即买入，五个交易日后收盘卖出。"""
    trades = []
    day_index = START_DAY_INDEX

    while day_index + HOLDING_DAYS < len(history_data):
        daily_score = calculate_score_for_day(
            history_data,
            day_index,
            temporary_directory,
        )

        if daily_score >= BUY_SCORE:
            trades.append(create_trade(stock_name, history_data, day_index))
            day_index += HOLDING_DAYS + 1
        else:
            day_index += 1

    return trades


def calculate_statistics(trades, stock_count):
    """根据全部交易记录计算回测报告需要的基础统计数据。"""
    if not trades:
        return {
            "股票数": stock_count,
            "交易次数": 0,
            "胜率": 0.0,
            "平均收益率": 0.0,
            "累计收益率": 0.0,
            "最大单笔收益": 0.0,
            "最大单笔亏损": 0.0,
        }

    return_rates = [trade["收益率"] for trade in trades]
    winning_trades = [return_rate for return_rate in return_rates if return_rate > 0]

    return {
        "股票数": stock_count,
        "交易次数": len(trades),
        "胜率": len(winning_trades) / len(trades) * 100,
        "平均收益率": sum(return_rates) / len(trades),
        # 第一版不模拟仓位，累计收益率为全部交易收益率的简单相加。
        "累计收益率": sum(return_rates),
        "最大单笔收益": max(return_rates),
        "最大单笔亏损": min(return_rates),
    }


def print_report(statistics):
    """按固定格式在终端输出 AStockAI 回测报告。"""
    print("==================")
    print("AStockAI 回测报告")
    print("==================")
    print(f"股票：{statistics['股票数']} 只")
    print(f"交易次数：{statistics['交易次数']}")
    print(f"胜率：{statistics['胜率']:.2f}%")
    print(f"平均收益率：{statistics['平均收益率']:.2f}%")
    print(f"累计收益率：{statistics['累计收益率']:.2f}%")
    print(f"最大单笔收益：{statistics['最大单笔收益']:.2f}%")
    print(f"最大单笔亏损：{statistics['最大单笔亏损']:.2f}%")
    print("==================")


def run_backtest():
    """执行 data 文件夹内所有历史行情文件的第一版基础回测。"""
    data_directory = Path(__file__).parent / "data"
    historical_files = find_historical_data_files(data_directory)
    all_trades = []

    with TemporaryDirectory() as temporary_path:
        temporary_directory = Path(temporary_path)

        for file_path in historical_files:
            history_data = load_historical_data(file_path)
            stock_name = get_stock_name(file_path)
            stock_trades = backtest_stock(
                stock_name,
                history_data,
                temporary_directory,
            )
            all_trades.extend(stock_trades)

    statistics = calculate_statistics(all_trades, len(historical_files))
    print_report(statistics)
    return all_trades, statistics


if __name__ == "__main__":
    run_backtest()
