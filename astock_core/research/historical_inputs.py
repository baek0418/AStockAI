"""可复核研究输入：逐日股票池、原始价格/交易状态、公司行为和交易日历。

只读取隔离研究目录；绝不把今天的股票池或前复权快照推定为历史证据。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_NAMES = ('calendar', 'membership', 'prices', 'corporate_actions')
ACTION_COLUMNS = ['事件编号','股票代码','登记日','除权日','派息日','新增股份上市日',
                  '每股税后红利','每股送转股数','公告时间','类型']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bool_series(values, label):
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    result = values.astype(str).str.lower().map(mapping)
    if result.isna().any():
        raise ValueError(f'{label}缺失或不是明确的 True/False，不能假定可交易。')
    return result.astype(bool)


def _dates(frame, columns):
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors='raise')
        if frame[column].isna().any():
            raise ValueError(f'{column}缺失。')


def load_historical_inputs(project_directory):
    root = Path(project_directory) / 'data/research_execution'
    manifest_file = root / 'manifest.json'
    if not manifest_file.is_file():
        raise ValueError('缺少 data/research_execution/manifest.json：不能用当前成分股、前复权价格代替历史股票池及原始成交证据。')
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get('version') != 1 or manifest.get('corporate_actions_complete') is not True:
        raise ValueError('研究输入版本无效，或尚未明确核对公司行为完整性。')
    frames, hashes = {}, {'manifest.json': sha256(manifest_file)}
    for name in INPUT_NAMES:
        spec = manifest.get('files', {}).get(name, {})
        path = (root / spec.get('path', '')).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or not spec.get('source'):
            raise ValueError(f'{name}缺少隔离目录内文件或来源。')
        digest = sha256(path)
        if digest != spec.get('sha256'):
            raise ValueError(f'{name}内容校验值不匹配。')
        hashes[name] = digest
        frames[name] = pd.read_csv(path, dtype={'股票代码': str})
    calendar = frames['calendar']
    _dates(calendar, ['日期'])
    if calendar['日期'].duplicated().any() or not calendar['日期'].is_monotonic_increasing:
        raise ValueError('交易日历重复或未按升序排列。')
    sessions = pd.DatetimeIndex(calendar['日期'])
    if len(sessions) < 2:
        raise ValueError('交易日历不足。')
    members = frames['membership']
    _dates(members, ['日期','可得时间'])
    if members.duplicated(['日期','股票代码']).any():
        raise ValueError('历史成分存在重复记录。')
    expected_size = manifest.get('universe_size')
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 1:
        raise ValueError('未声明逐日完整股票池数量。')
    counts = members.groupby('日期').size().reindex(sessions)
    if counts.isna().any() or not counts.eq(expected_size).all() or not members['日期'].isin(sessions).all():
        raise ValueError('逐日历史股票池不完整；不允许跨缺失日期沿用最新成分。')
    if (members['可得时间'] >= members['日期'] + pd.Timedelta(days=1)).any():
        raise ValueError('股票池包含信号日之后才可得的成分记录。')
    prices = frames['prices']
    required = {'日期','股票代码','收盘','成交量','复权方式','可买入','可卖出','可得时间','状态来源'}
    if not required.issubset(prices):
        raise ValueError(f'原始价格/交易状态缺少字段：{sorted(required-set(prices))}')
    _dates(prices, ['日期','可得时间'])
    if prices.duplicated(['日期','股票代码']).any() or not prices['日期'].isin(sessions).all():
        raise ValueError('原始价格重复或不在日历内。')
    if not prices['复权方式'].eq('raw').all():
        raise ValueError('交易执行只能使用原始价格，禁止前复权价格计算股数和成交金额。')
    if (prices['可得时间'] >= prices['日期'] + pd.Timedelta(days=1)).any():
        raise ValueError('价格或交易状态在当日收盘后信号截止时间仍不可得。')
    if prices['状态来源'].isna().any() or prices['状态来源'].astype(str).str.strip().eq('').any():
        raise ValueError('交易状态缺少来源。')
    for column in ('可买入','可卖出'):
        prices[column] = _bool_series(prices[column],column)
    for column in ('收盘','成交量'):
        prices[column] = pd.to_numeric(prices[column],errors='raise')
    if not np.isfinite(prices[['收盘','成交量']]).all().all() or prices['收盘'].le(0).any() or prices['成交量'].lt(0).any():
        raise ValueError('原始价格或成交量无效；停牌估值也必须有显式来源记录。')
    for frame in (members, prices):
        if frame['股票代码'].isna().any() or not frame['股票代码'].str.fullmatch(r'\d{6}').all():
            raise ValueError('历史输入必须用六位股票代码作为稳定标识。')
    missing_codes = set(members['股票代码'])-set(prices['股票代码'])
    if missing_codes:
        raise ValueError(f'历史成分缺少价格文件，拒绝删股：{sorted(missing_codes)}')
    missing_quotes = members[['日期','股票代码']].merge(prices[['日期','股票代码']],on=['日期','股票代码'],how='left',indicator=True)
    if missing_quotes['_merge'].eq('left_only').any():
        raise ValueError('历史成分在当日缺少原始价格/状态；拒绝整个实验，不按数据完整性删股。')
    actions = frames['corporate_actions']
    if not set(ACTION_COLUMNS).issubset(actions):
        raise ValueError('公司行为台账字段不完整；无事件也须提供带表头空表及完整性声明。')
    if not actions.empty:
        _dates(actions, ['登记日','除权日','公告时间'])
        for column in ('派息日','新增股份上市日'):
            actions[column] = pd.to_datetime(actions[column], errors='coerce')
        if actions['事件编号'].isna().any() or actions['事件编号'].duplicated().any():
            raise ValueError('公司行为事件编号缺失或重复。')
        if not actions['类型'].eq('distribution').all():
            raise ValueError('存在配股、合并、退市结算等未支持的公司行为，拒绝自动近似。')
        for column in ('每股税后红利','每股送转股数'):
            actions[column] = pd.to_numeric(actions[column],errors='raise')
            if not np.isfinite(actions[column]).all() or actions[column].lt(0).any():
                raise ValueError('公司行为分配金额/比例无效。')
        if (actions['公告时间'] >= actions['登记日'] + pd.Timedelta(days=1)).any():
            raise ValueError('公司行为公告晚于登记日。')
        next_day = dict(zip(sessions[:-1],sessions[1:]))
        if not actions['登记日'].map(next_day).eq(actions['除权日']).all():
            raise ValueError('只支持登记日下一交易日除权的分配；其他事件需单独核验。')
        cash_events = actions['每股税后红利'].gt(0)
        stock_events = actions['每股送转股数'].gt(0)
        if actions.loc[cash_events, '派息日'].isna().any() or (actions.loc[cash_events, '派息日'] < actions.loc[cash_events, '除权日']).any():
            raise ValueError('现金分红必须有不早于除权日的派息日。')
        if actions.loc[stock_events, '新增股份上市日'].isna().any() or (actions.loc[stock_events, '新增股份上市日'] < actions.loc[stock_events, '除权日']).any():
            raise ValueError('送转股必须有不早于除权日的新增股份上市日。')
        if not set(actions['股票代码']).issubset(set(prices['股票代码'])):
            raise ValueError('公司行为股票缺少原始价格。')
    audit = {'状态':'validated_inputs','输入校验值':hashes,'股票池口径':'逐交易日完整成分，按信号日可得时间筛选',
             '价格口径':'原始价格成交；特征使用逐日公司行为构造的因果收益序列',
             '交易状态':'显式双向交易许可；缺失拒绝，不由OHLCV猜测',
             '公司行为':'登记日权益、除权日应收红利、派息日现金、上市日解锁新增股份',
             '日期范围':[str(sessions[0].date()),str(sessions[-1].date())],
             '限制':['来源与完整性由输入台账声明，校验值不等于独立证明历史从未修订。',
                     '股息税采用输入的税后金额情景；不模拟按持有期追缴税。',
                     '收盘价加固定滑点是成交假设，不证明实际可成交数量。'],
             '是否策略准入':False}
    return frames, audit


def build_historical_features(frames):
    """公司行为只在当日进入收益序列；先计算特征/标签，再按历史成分筛信号。"""
    from astock_core.research.prediction_features import build_stock_feature_frame
    sessions = pd.DatetimeIndex(frames['calendar']['日期'])
    actions = frames['corporate_actions']
    parts = []
    for code, raw in frames['prices'].groupby('股票代码'):
        raw = raw.set_index('日期').reindex(sessions)
        distributions = actions[actions['股票代码'].eq(code)]
        cash = distributions.groupby('除权日')['每股税后红利'].sum() if not distributions.empty else pd.Series(dtype=float)
        bonus = distributions.groupby('除权日')['每股送转股数'].sum() if not distributions.empty else pd.Series(dtype=float)
        # 缺行情时重新开始历史窗口，绝不填造跨缺口收益或将5根K线当5个全市场交易日。
        groups = raw['收盘'].isna().cumsum()
        for _, segment in raw.groupby(groups):
            segment = segment.dropna(subset=['收盘']).copy()
            if segment.empty:
                continue
            close = segment['收盘']
            returns = (close*(1+bonus.reindex(segment.index,fill_value=0)) + cash.reindex(segment.index,fill_value=0))/close.shift(1)
            returns.iloc[0] = 1.0
            segment['收盘'] = close.iloc[0] * returns.cumprod()
            history = segment[['收盘','成交量']].rename_axis('日期').reset_index()
            parts.append(build_stock_feature_frame(history,code,code))
    features = pd.concat(parts,ignore_index=True)
    # 与现有OHLCV基线保持相同特征列；市场基准由调用方对齐。
    eligible = frames['membership'][['日期','股票代码']]
    return features.merge(eligible,on=['日期','股票代码'],how='inner',validate='one_to_one')
