import os
import pandas as pd

from score import calculate_score


# =====================
# 股票池
# =====================

stocks = [
    "贵州茅台",
    "宁德时代",
    "招商银行",
    "五粮液",
    "平安银行",
    "美的集团",
    "中国平安",
    "比亚迪",
    "格力电器",
    "隆基绿能"
]


results = []


# =====================
# 扫描
# =====================

for stock in stocks:


    file = f"data/{stock}历史.csv"


    if not os.path.exists(file):

        print(
            stock,
            "没有数据，跳过"
        )

        continue



    try:

        result = calculate_score(file)


        results.append(
            {
                "股票":stock,
                **result
            }
        )


    except Exception as e:

        print(
            stock,
            "失败:",
            e
        )



# =====================
# 排序
# =====================

df = pd.DataFrame(results)


df = df.sort_values(
    by="评分",
    ascending=False
)



print("===================")

print(
    "AStockAI 股票扫描报告"
)

print("===================")


print(
    df[
        [
            "股票",
            "评分",
            "RSI",
            "MA5",
            "MA20",
            "建议"
        ]
    ]
    .to_string(
        index=False
    )
)