import pandas as pd


def calculate_score_dataframe(dataframe):
    """对已有历史日线 DataFrame 使用原有 RSI、MA、MACD 与评分规则。"""
    df = dataframe.copy()


    # =====================
    # 基础指标
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


    df["RSI"] = (
        100 -
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


    trend = (
        latest["MA5"]
        -
        latest["MA20"]
    ) / latest["MA20"]


    if trend > 0.03:

        score += 30

    elif trend > 0:

        score += 20

    elif trend > -0.03:

        score += 10

    else:

        score += 0



    # =====================
    # 2 RSI 25分
    # =====================


    rsi = latest["RSI"]


    if 50 <= rsi <= 65:

        score += 25


    elif 40 <= rsi < 50:

        score += 15


    elif 30 <= rsi < 40:

        score += 10


    elif rsi > 80:

        score -= 10



    # =====================
    # 3 MACD 20分
    # =====================


    macd = latest["MACD"]


    if macd > 0:

        score += 15


    if (
        df["MACD"].iloc[-1]
        >
        df["MACD"].iloc[-2]
    ):

        score += 5



    # =====================
    # 4 成交量 15分
    # =====================


    vol20 = (
        df["成交量"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )


    if latest["成交量"] > vol20:

        score += 15

    elif latest["成交量"] > vol20*0.8:

        score += 8



    # =====================
    # 5 五日涨幅 10分
    # =====================


    change5 = (

        latest["收盘"]
        -
        df["收盘"].iloc[-6]

    ) / df["收盘"].iloc[-6]


    if change5 > 0.05:

        score += 10


    elif change5 > 0:

        score += 5



    # 防止超过100

    score = max(
        0,
        min(
            100,
            int(score)
        )
    )



    # 建议

    if score >= 80:

        advice="强烈关注"

    elif score >=65:

        advice="重点观察"

    elif score>=50:

        advice="观望"

    else:

        advice="风险"



    return {

        "评分":score,

        "RSI":round(
            latest["RSI"],2
        ),

        "MACD":round(
            latest["MACD"],2
        ),

        "MA5":round(
            latest["MA5"],2
        ),

        "MA20":round(
            latest["MA20"],2
        ),

        "建议":advice

    }


def calculate_score(file):
    """读取 CSV 后使用原有评分规则，保持既有调用方式兼容。"""
    return calculate_score_dataframe(pd.read_csv(file))
