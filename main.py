import requests
import pandas as pd
import os


stocks = {
    "sh600519": "贵州茅台",
    "sz000001": "平安银行",
    "sh600036": "招商银行",
    "sz000858": "五粮液",
    "sz300750": "宁德时代"
}


for code, name in stocks.items():

    url = f"https://qt.gtimg.cn/q={code}"

    response = requests.get(url)

    data = response.text.split('"')[1]

    items = data.split("~")


    stock = {
        "股票名称": items[1],
        "股票代码": items[2],
        "当前价格": float(items[3]),
        "昨收": float(items[4]),
        "开盘": float(items[5]),
        "成交量": items[6],
        "成交额": items[7],
        "涨跌额": float(items[31]),
        "涨跌幅": float(items[32]),
    }


    df = pd.DataFrame([stock])


    filename = f"data/{name}.csv"

    df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )


    print(name, "保存成功")