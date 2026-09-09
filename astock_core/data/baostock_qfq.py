"""BaoStock 前复权适配：子进程隔离会话并限制网络等待时间。"""
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

FIELDS = 'date,code,open,high,low,close,volume,adjustflag,tradestatus'


def normalize_response(rows, market_code):
    from astock_core.data.market_data_sources import normalize_history
    code = market_code[:2] + '.' + market_code[2:]
    frame = pd.DataFrame(rows, columns=FIELDS.split(','))
    if frame.empty or not frame['code'].eq(code).all() or not frame['adjustflag'].eq('2').all():
        raise ValueError('BaoStock 返回代码或前复权标志不匹配，拒绝使用。')
    frame = frame.rename(columns={'date':'日期','open':'开盘','close':'收盘','high':'最高','low':'最低','volume':'成交量'})
    # BaoStock 为股，正式腾讯日线为手，统一为每手 100 股。
    frame['成交量'] = pd.to_numeric(frame['成交量'], errors='raise') / 100
    return normalize_history(frame).tail(601).reset_index(drop=True)


class BaoStockQfqDailySource:
    name = 'BaoStock qfq kline'
    adjustment = 'qfq'
    source_note = 'BaoStock 0.9.3；adjustflag=2，逐行核验代码与复权标志；成交量由股转换为手；完整序列替换，不跨来源拼接。'

    def fetch(self, market_code):
        if len(market_code) != 8 or market_code[:2] not in ('sh','sz') or not market_code[2:].isdigit():
            raise ValueError('无效的股票市场代码。')
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'astock_core.data.baostock_qfq', market_code],
                capture_output=True, text=True, timeout=25, cwd=Path(__file__).parents[2],
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError('BaoStock 前复权查询超时（25秒），已终止隔离进程。') from error
        if result.returncode:
            raise ValueError('BaoStock 前复权查询失败：' + result.stderr.strip()[-500:])
        return normalize_response(json.loads(result.stdout), market_code)


def main():
    import baostock as bs
    code = sys.argv[1]
    with contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
        if login.error_code != '0':
            raise ValueError(f'BaoStock 登录失败：{login.error_msg}')
        try:
            result = bs.query_history_k_data_plus(
                code[:2]+'.'+code[2:], FIELDS, start_date=(date.today() - timedelta(days=1100)).isoformat(),
                end_date=date.today().isoformat(), frequency='d', adjustflag='2',
            )
            if result.error_code != '0':
                raise ValueError(result.error_msg)
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            if result.error_code != '0':
                raise ValueError(result.error_msg)
        finally:
            bs.logout()
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
