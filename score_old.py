import pandas as pd
import os


def calculate_score(file):

    df = pd.read_csv(
        file
    )


    # =====================
    # 计算指标
    # =====================

    df["MA5"] = (
        df["收盘"]
        .rolling(5)
        .mean()
    )


    df["MA20"] = (
        df["收盘"]
        .rolling(20)
        .mean()
    )


    # RSI

    delta = df["收盘"].diff()

    gain = (
        delta
        .where(delta > 0,0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .where(delta < 0,0)
        .rolling(14)
        .mean()
    )


    rs = gain / loss

    df["RSI"] = 100 - (
        100/(1+rs)
    )


    # MACD

    ema12 = (
        df["收盘"]
        .ewm(span=12)
        .mean()
    )

    ema26 = (
        df["收盘"]
        .ewm(span=26)
        .mean()
    )


    df["MACD"] = ema12-ema26



    latest = df.iloc[-1]


    score = 0



    # =====================
    # 1 趋势 30分
    # =====================


    if latest["收盘"] > latest["MA5"]:
        score += 10


    if latest["MA5"] > latest["MA20"]:
        score += 10


    if latest["收盘"] > latest["MA20"]:
        score += 10



    # =====================
    # 2 RSI 25分
    # =====================

    rsi = latest["RSI"]


    if 40 <= rsi <= 70:
        score += 15


    if 50 <= rsi <= 60:
        score += 10



    # =====================
    # 3 MACD 25分
    # =====================

    if latest["MACD"] > 0:

        score += 15


    if df["MACD"].iloc[-1] > df["MACD"].iloc[-2]:

        score += 10



    # =====================
    # 4 成交量 20分
    # =====================


    vol5 = (
        df["成交量"]
        .rolling(5)
        .mean()
        .iloc[-1]
    )


    if latest["成交量"] > vol5:

        score += 20



    # =====================
    # 建议
    # =====================


    if score >=80:

        advice="强烈关注"

    elif score>=65:

        advice="重点观察"

    elif score>=50:

        advice="观望"

    else:

        advice="回避"



    return {

        "评分":score,

        "RSI":round(
            latest["RSI"],2
        ),

        "MACD":round(
            latest["MACD"],2
        ),

        "建议":advice
    }