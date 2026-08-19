import requests
import pandas as pd


# 贵州茅台
code = "sh600519"


url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,500,qfq"


response = requests.get(url)


data = response.json()


# 提取K线
kline = data["data"][code]["qfqday"]


df = pd.DataFrame(
    kline,
    columns=[
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额"
    ]
)


print(df.head())

print("数据数量:")
print(len(df))


# 保存
df.to_csv(
    "data/贵州茅台历史.csv",
    index=False,
    encoding="utf-8-sig"
)


print("历史数据保存成功")