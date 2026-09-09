import json
import subprocess
import unittest
from unittest.mock import patch, Mock
from astock_core.data.baostock_qfq import normalize_response, BaoStockQfqDailySource
from astock_core.data.market_data_sources import default_daily_sources


class BaoStockQfqTests(unittest.TestCase):
    def rows(self):
        return [['2026-09-08','sz.000002','3.2','3.3','3.1','3.27','168673162','2','1']]

    def test_validated_qfq_has_unified_volume_units(self):
        history = normalize_response(self.rows(),'sz000002')
        self.assertEqual(history.iloc[0]['收盘'],3.27)
        self.assertAlmostEqual(history.iloc[0]['成交量'],1686731.62)

    def test_wrong_code_or_raw_flag_rejected(self):
        for column,value in [(1,'sh.600000'),(7,'3')]:
            rows=self.rows()
            rows[0][column]=value
            with self.assertRaises(ValueError):
                normalize_response(rows,'sz000002')

    def test_hung_worker_is_bounded(self):
        with patch('astock_core.data.baostock_qfq.subprocess.run',side_effect=subprocess.TimeoutExpired('test',25)):
            with self.assertRaisesRegex(ValueError,'超时'):
                BaoStockQfqDailySource().fetch('sz000002')

    def test_worker_result_is_validated(self):
        result=Mock(returncode=0,stdout=json.dumps(self.rows()))
        with patch('astock_core.data.baostock_qfq.subprocess.run',return_value=result) as run:
            frame=BaoStockQfqDailySource().fetch('sz000002')
        self.assertEqual(len(frame),1)
        self.assertEqual(run.call_args.kwargs['timeout'],25)

    def test_formal_sources_only_use_qfq(self):
        sources=default_daily_sources()
        self.assertEqual([s.adjustment for s in sources],['qfq','qfq'])
        self.assertIsInstance(sources[1],BaoStockQfqDailySource)
