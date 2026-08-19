import pandas as pd
import ta
import os


# 股票池

stocks = [
    "贵州茅台",
    "宁德时代",
    "招商银行",
    "五粮液",
    "平安银行"
]


results = []


for stock in stocks:

    file = f"data/{stock}历史.csv"

    # 如果没有数据跳过

    if not os.path.exists(file):
        print(stock, "没有数据，跳过")
        continue


    df = pd.read_csv(file)


    df = df.sort_values("日期")


    close = df["收盘"]


    # 均线

    ma5 = close.rolling(5).mean().iloc[-1]

    ma20 = close.rolling(20).mean().iloc[-1]


    # RSI

    rsi = ta.momentum.RSIIndicator(
        close
    ).rsi().iloc[-1]


    # MACD

    macd = ta.trend.MACD(
        close
    ).macd().iloc[-1]


    # 评分

    score = 50


    if ma5 > ma20:
        score += 20


    if rsi < 30:
        score += 15

    elif rsi > 70:
        score -= 15


    if macd > 0:
        score += 15



    # 建议

    if score >=80:
        advice="强势"

    elif score>=60:
        advice="偏多"

    else:
        advice="观望"



    results.append(
        [
            stock,
            round(score),
            round(rsi,2),
            round(macd,2),
            advice
        ]
    )



# 转成表格

result_df = pd.DataFrame(
    results,
    columns=[
        "股票",
        "评分",
        "RSI",
        "MACD",
        "建议"
    ]
)


# 排序

result_df = result_df.sort_values(
    "评分",
    ascending=False
)


print("===================")
print("AStockAI 股票扫描报告")
print("===================")

print(result_df.to_string(index=False))