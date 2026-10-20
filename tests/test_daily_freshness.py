import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd
import requests

from astock_core.data.daily_freshness import latest_completed_session, completed_history
from astock_core.data.prediction_benchmark_data import fetch_tencent_index_daily
from astock_core.data.market_data_sources import normalize_history
from astock_core.pipeline import get_pipeline_status, save_pipeline_status


class DailyFreshnessTests(unittest.TestCase):
    def test_intraday_and_weekend_use_last_completed_session(self):
        self.assertEqual(latest_completed_session(datetime(2026, 9, 9, 14)), '2026-09-08')
        self.assertEqual(latest_completed_session(datetime(2026, 9, 9, 16)), '2026-09-09')
        self.assertEqual(latest_completed_session(datetime(2026, 9, 13, 16)), '2026-09-11')
        self.assertEqual(latest_completed_session(datetime(2026, 10, 8, 9)), '2026-09-30')

    def test_unknown_calendar_fails_closed(self):
        with self.assertRaisesRegex(ValueError, '交易日历'):
            latest_completed_session(datetime(2027, 6, 1, 16))

    def test_partial_bar_removed_and_stale_response_rejected(self):
        frame = pd.DataFrame({'日期': ['2026-09-07', '2026-09-08', '2026-09-09'], '收盘': [10, 11, 12]})
        self.assertEqual(len(completed_history(frame, '2026-09-08')), 2)
        with self.assertRaisesRegex(ValueError, '保留旧文件'):
            completed_history(frame.iloc[:1], '2026-09-08')

    def test_infinite_prices_not_accepted(self):
        frame = pd.DataFrame([['2026-09-08', 10, 10, float('inf'), 9, 100]],
                             columns=['日期', '开盘', '收盘', '最高', '最低', '成交量'])
        with self.assertRaises(ValueError):
            normalize_history(frame)

    def test_benchmark_retries_transient_failure(self):
        response = Mock()
        response.json.return_value = {'data': {'sh000300': {'day': [['2026-09-08', '1', '1', '1', '1', '1']]}}}
        getter = Mock(side_effect=[requests.Timeout('timeout'), response])
        with patch('astock_core.data.prediction_benchmark_data.time.sleep'):
            self.assertEqual(len(fetch_tencent_index_daily('sh000300', getter)), 1)
        self.assertEqual(getter.call_count, 2)

    def test_failed_benchmark_cannot_make_pipeline_success(self):
        self.assertEqual(get_pipeline_status({'success': True, 'steps': [{'status': 'success'}, {'status': 'failed'}]}), 'partial')

    def test_pipeline_status_saved_at_project_root_after_module_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('astock_core.pipeline.__file__', str(root / 'astock_core' / 'pipeline.py')):
                output = save_pipeline_status({'run_id': 'test', 'started_at': '', 'finished_at': '',
                                              'status': 'success', 'duration': 0, 'steps': []})
            self.assertEqual(output, root / 'output/pipeline_status.json')
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))['run_id'], 'test')

    def test_default_sources_have_no_eastmoney_or_raw_fallback(self):
        from astock_core.data.market_data_sources import default_daily_sources, default_raw_daily_sources
        self.assertTrue(all(source.adjustment == 'qfq' for source in default_daily_sources()))
        self.assertTrue(all('eastmoney' not in getattr(source, 'url', '') for source in (*default_daily_sources(), *default_raw_daily_sources())))

    def test_non_qfq_batch_preserves_existing_files(self):
        from dataclasses import replace
        from test_update_data_batch import make_result
        from astock_core.data.update_data import run_update_data
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            (data / 'provenance').mkdir(parents=True)
            history = data / '甲历史.csv'
            provenance = data / 'provenance' / '000001.json'
            history.write_bytes(b'old history')
            provenance.write_bytes(b'old provenance')
            for adjustment in ('raw', 'unknown', 'hfq'):
                with self.subTest(adjustment=adjustment):
                    result = run_update_data(directory, [{'code': '000001', 'name': '甲', 'market_code': 'sz000001'}],
                                             fetch_history=lambda code: replace(make_result(), adjustment=adjustment))
                    self.assertEqual(result['status'], 'failed')
                    self.assertEqual(result['details']['failed_count'], 1)
                    self.assertIn('前复权', result['details']['failed_stocks'][0]['error'])
                    self.assertEqual(history.read_bytes(), b'old history')
                    self.assertEqual(provenance.read_bytes(), b'old provenance')
