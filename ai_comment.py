def generate_comment(
        stock,
        score,
        rsi,
        macd,
        ma5,
        ma20
):

    result = []

    result.append(
        f"{stock} 综合评分：{score}/100"
    )


    result.append("\n【趋势分析】")


    if ma5 > ma20:

        result.append(
            "✅ 短期均线强于中期均线，趋势偏向上。"
        )

    else:

        result.append(
            "⚠ 短期均线弱于中期均线，趋势偏弱。"
        )



    result.append("\n【动量分析】")


    if rsi >= 70:

        result.append(
            "⚠ RSI较高，短线可能存在过热风险。"
        )

    elif rsi >= 50:

        result.append(
            "✅ RSI处于健康区间，市场动能较稳定。"
        )

    else:

        result.append(
            "📉 RSI偏低，可能处于弱势或超跌阶段。"
        )



    result.append("\n【MACD分析】")


    if macd > 0:

        result.append(
            "✅ MACD为正，短期动能较强。"
        )

    else:

        result.append(
            "⚠ MACD为负，趋势动能仍需观察。"
        )



    result.append("\n【综合建议】")


    if score >= 80:

        result.append(
            "🔥 强势标的，可重点关注。"
        )

    elif score >= 65:

        result.append(
            "👀 当前技术面较好，建议持续观察。"
        )

    elif score >= 50:

        result.append(
            "⏳ 走势中性，等待趋势确认。"
        )

    else:

        result.append(
            "⚠ 当前信号较弱，注意风险。"
        )


    return "\n".join(result)