"""核验正式股票池和基准的本地完整日线，不请求网络。"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from astock_core.data.daily_freshness import completed_history, latest_completed_session
from astock_core.data.stock_universe import get_enabled_stock_universe
from astock_core.data.prediction_benchmark_data import BENCHMARKS


def run_audit():
    root = Path(__file__).parents[2]
    expected = latest_completed_session()
    records = []
    for stock in get_enabled_stock_universe():
        record = {'code': stock['code'], 'name': stock['name']}
        try:
            frame = pd.read_csv(root / 'data' / f"{stock['name']}历史.csv")
            completed_history(frame, expected)
            metadata = json.loads((root / 'data/provenance' / f"{stock['code']}.json").read_text(encoding="utf-8"))
            if metadata['日期范围'][-1] != expected:
                raise ValueError('来源记录日期与完整交易日不一致')
            record.update(status='success', adjustment=metadata['复权方式'], source=metadata['数据源'], latest_date=expected)
        except (OSError, ValueError, KeyError) as error:
            record.update(status='failed', error=str(error))
        records.append(record)
    benchmarks = []
    for name, definition in BENCHMARKS.items():
        try:
            frame = pd.read_csv(root / 'data/market' / definition['file_name'])
            completed_history(frame, expected)
            benchmarks.append({'name': name, 'status': 'success', 'latest_date': expected})
        except (OSError, ValueError, KeyError) as error:
            benchmarks.append({'name': name, 'status': 'failed', 'error': str(error)})
    failed = sum(r['status'] != 'success' for r in records)
    raw = sum(r.get('adjustment') == 'raw' for r in records)
    result = dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), expected_date=expected,
                  status='partial' if failed or raw or any(r['status'] != 'success' for r in benchmarks) else 'success',
                  stock_count=len(records), current_count=len(records)-failed, failed_count=failed,
                  raw_count=raw, stocks=records, benchmarks=benchmarks)
    target = root / 'output/data_production_status.json'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding="utf-8")
    temporary.replace(target)
    return result


if __name__ == '__main__':
    result = run_audit()
    print(json.dumps({k:v for k,v in result.items() if k != 'stocks'},ensure_ascii=False,indent=2))
