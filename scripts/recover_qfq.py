"""仅恢复正式池中标为 raw 的文件；隔离下载、核验、备份后整体替换。"""
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
from astock_core.data.baostock_qfq import BaoStockQfqDailySource
from astock_core.data.daily_freshness import completed_history, latest_completed_session
from astock_core.data.market_data_sources import fetch_daily_history
from astock_core.data.stock_universe import get_enabled_stock_universe
from astock_core.data.update_data import _save_stock_record


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'output/recovery' / datetime.now().strftime('%Y%m%d_%H%M%S')
    staging = folder / 'staging'
    staging.mkdir(parents=True)
    expected = latest_completed_session()
    records = []
    for stock in get_enabled_stock_universe():
        metadata_file = root / 'data/provenance' / f"{stock['code']}.json"
        if not metadata_file.exists() or json.loads(metadata_file.read_text()).get('复权方式') != 'raw':
            continue
        record = {'code':stock['code'],'name':stock['name']}
        try:
            original = root / 'data' / f"{stock['name']}历史.csv"
            old = pd.read_csv(original)
            result = fetch_daily_history(stock['market_code'], sources=[BaoStockQfqDailySource()])
            history = completed_history(result.history, expected)
            if len(history) < min(len(old), 590):
                raise ValueError('替代日线长度不足，拒绝截断旧历史。')
            old_end = old.loc[pd.to_datetime(old['日期']) == pd.Timestamp(expected), '收盘']
            if old_end.empty or abs(float(history['收盘'].iloc[-1]) / float(old_end.iloc[-1]) - 1) > .02:
                raise ValueError('最新前复权与原始收盘偏差超过2%，需人工核对公司行为。')
            result = replace(result,history=history)
            _save_stock_record(stock, staging, lambda _:result)
            record.update(status='validated',rows=len(history),latest_date=expected)
        except Exception as error:
            record.update(status='failed',error=str(error))
        records.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    # 所有下载完成后，才替换已验证的正式文件。旧数据与来源一并备份。
    for record in records:
        if record['status'] != 'validated':
            continue
        relative_files = [Path(f"{record['name']}历史.csv"),Path('provenance') / f"{record['code']}.json"]
        for relative in relative_files:
            original = root / 'data' / relative
            backup = folder / 'backup' / relative
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,backup)
        for relative in relative_files:
            original = root / 'data' / relative
            temporary = original.with_suffix(original.suffix + '.recovery.tmp')
            shutil.copy2(staging / relative,temporary)
            temporary.replace(original)
        record['status'] = 'restored'
    manifest = {'expected_date':expected,'stocks':records,'restored':sum(r['status']=='restored' for r in records),'failed':sum(r['status']=='failed' for r in records)}
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('MANIFEST',folder/'manifest.json',flush=True)
    return int(manifest['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
