import os
from datetime import datetime

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
# 获取评分
# =====================

for stock in stocks:


    file = f"data/{stock}历史.csv"


    if not os.path.exists(file):
        continue


    try:

        result = calculate_score(file)

        result["股票"] = stock

        results.append(result)


    except Exception as e:

        print(
            stock,
            "失败:",
            e
        )



# 排序

results = sorted(
    results,
    key=lambda x:x["评分"],
    reverse=True
)



today = datetime.now().strftime(
    "%Y-%m-%d"
)



filename = (
    f"AStockAI晨报_{today}.md"
)



with open(
    filename,
    "w",
    encoding="utf-8"
) as f:


    f.write(
f"""
# 📈 AStockAI 每日投资晨报

日期：{today}


======================


"""
    )


    # 强势池

    f.write(
"""
# 🔥 强势观察池


"""
    )


    for i,item in enumerate(results[:5],1):


        f.write(
f"""
## {i}. {item['股票']}


评分：

{item['评分']} /100


RSI：

{item['RSI']}


均线：

MA5 {item['MA5']}

MA20 {item['MA20']}


建议：

{item['建议']}


-------------------

"""
        )



    # 风险池

    f.write(
"""
# ⚠ 风险观察池


"""
    )


    for item in results[-3:]:


        f.write(
f"""
## {item['股票']}


评分：

{item['评分']}/100


建议：

{item['建议']}


-------------------

"""
        )



    # 总结

    strong = results[0]["股票"]

    weak = results[-1]["股票"]


    f.write(
f"""

# 🧠 AI市场总结


今日评分最高：

{strong}


当前风险最高：

{weak}


系统建议：

关注高评分股票，
规避评分持续低于40的标的。


"""
    )



print("===================")

print(
    "晨报生成成功:"
)

print(filename)

print("===================")