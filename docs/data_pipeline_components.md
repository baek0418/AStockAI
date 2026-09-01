# 行情、指标与研究过程组件

本项目将行情更新、技术指标、模型训练与研究评估保持为独立模块。它们服务于
本地研究，不构成交易指令，也不会绕过既有 v5.0/v5.1 的样本外验证准入。

## 日线主备源与全市场更新

运行 `python update_data.py`，或从 Web 的“后台更新本地行情”触发更新。

- 主源为腾讯 `qfqday` 前复权日线；仅当主源无法返回有效 OHLCV 时，才将该股票
  完整切换至东方财富前复权日线。
- 不会将两家来源的行拼接到同一只股票中；若两者都失败，将保留旧 CSV 并记录失败，
  不会写入空文件。
- 默认以 3 个工作线程处理股票池，并在所有线程间保持至少 0.35 秒的请求间隔。
  单个来源遇到暂时性失败会指数退避重试最多 3 次，再切换整只股票到备用源。每只成功更新的股票会在
  `data/provenance/<六位代码>.json` 写入来源、复权方式、日期范围及主源失败记录。
- 正式日线仍只写入 `data/*.csv`。审计文件位于子目录，不会被现有研究或回测误读为
  股票历史数据。

## 技术指标模块

`technical_indicators.py` 统一按日期升序、只使用当日与过去 OHLCV 计算 RSI、
MACD、KDJ、布林带、动量、波动率与量能相对值。扩展指标会作为
`量化事实快照` 中每只股票的 `扩展技术指标` 保存，用于解释和后续独立实验。

这些指标当前**没有**自动加入 v5.0/v5.1 的模型特征列表。因此既有模型目标、特征
定义、时间 gap、校准和样本外结果均未改变。若后续要将任一指标加入模型，必须新建
模型版本，并重新完成滚动样本外验证和成本后组合回测。

## 过程日志与模型边界

- 全市场更新事件：`logs/market_data_update_YYYY-MM-DD.jsonl`
- v5.0 训练事件：`logs/prediction_v50_training_YYYY-MM-DD.jsonl`
- v5.1 训练事件：`logs/prediction_v51_training_YYYY-MM-DD.jsonl`

JSONL 会记录阶段、状态、样本数、跳过文件数、来源使用情况和输出路径；不写入 API
令牌或环境变量。模型训练继续由 `prediction_model.py` 管理，特征由
`prediction_features.py`/`prediction_features_v2.py` 管理，评估与训练入口分别保留在
`prediction_evaluation.py`、`train_prediction.py` 和 `train_prediction_v2.py`。
