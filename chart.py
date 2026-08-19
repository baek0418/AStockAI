import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = ['Arial Unicode MS']


# 读取历史数据

df = pd.read_csv(
    "data/贵州茅台历史.csv"
)


# 转换日期

df["日期"] = pd.to_datetime(df["日期"])


# 按日期排序

df = df.sort_values("日期")


# 计算均线

df["MA5"] = df["收盘"].rolling(5).mean()

df["MA20"] = df["收盘"].rolling(20).mean()


# 画图

plt.figure(figsize=(12,6))


plt.plot(
    df["日期"],
    df["收盘"],
    label="Close Price"
)


plt.plot(
    df["日期"],
    df["MA5"],
    label="MA5"
)


plt.plot(
    df["日期"],
    df["MA20"],
    label="MA20"
)


plt.title("贵州茅台 Stock Trend")

plt.xlabel("Date")

plt.ylabel("Price")


plt.legend()

plt.grid()


plt.show()