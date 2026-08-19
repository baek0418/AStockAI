import pandas as pd


# 读取贵州茅台数据
df = pd.read_csv(
    "data/贵州茅台.csv"
)


print(df)


# 计算5日均线
df["MA5"] = df["当前价格"].rolling(5).mean()


# 计算20日均线
df["MA20"] = df["当前价格"].rolling(20).mean()


print(df)