import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from astock_core.research.historical_package_builder import audit_probe_directory, build_strict_package


class HistoricalPackageBuilderTests(unittest.TestCase):
    def make_candidate(self, root):
        source = root / 'candidate'; source.mkdir()
        dates = pd.to_datetime(['2025-01-02','2025-01-03'])
        pd.DataFrame({'日期': dates}).to_csv(source/'calendar.csv', index=False)
        pd.DataFrame({'日期': dates, '股票代码':['000001']*2,
                      '可得时间':['2025-01-02 16:00','2025-01-03 16:00']}).to_csv(source/'membership.csv',index=False)
        pd.DataFrame({'日期':dates,'股票代码':['000001']*2,'收盘':[10,11],
                      '成交量':[1000,1100],'adjustflag':['3','3']}).to_csv(source/'prices_raw.csv',index=False)
        pd.DataFrame({'日期':dates,'股票代码':['000001']*2,'可买入':[True,True],
                      '可卖出':[True,True],'可得时间':['2025-01-02 16:00','2025-01-03 16:00'],
                      '状态来源':['核验状态']*2}).to_csv(source/'execution_states.csv',index=False)
        pd.DataFrame(columns=['事件编号','股票代码','登记日','除权日','派息日','新增股份上市日',
                              '每股税后红利','每股送转股数','公告时间','类型']).to_csv(source/'corporate_actions.csv',index=False)
        evidence={'membership_availability_verified':True,'execution_states_complete':True,
                  'corporate_actions_complete':True,'universe_size':1,
                  'sources':{key:'合成测试证据' for key in ('calendar','membership','prices','corporate_actions')}}
        (source/'evidence.json').write_text(json.dumps(evidence), encoding="utf-8")
        return source

    def test_publishes_only_after_formal_loader_accepts_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=self.make_candidate(root)
            result=build_strict_package(root,source)
            self.assertEqual(result['status'],'published')
            self.assertTrue((root/'data/research_execution/manifest.json').is_file())

    def test_missing_evidence_never_creates_formal_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=self.make_candidate(root)
            evidence=json.loads((source/'evidence.json').read_text(encoding="utf-8")); evidence['execution_states_complete']=False
            (source/'evidence.json').write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaisesRegex(ValueError,'证据声明未通过'):
                build_strict_package(root,source)
            self.assertFalse((root/'data/research_execution').exists())

    def test_probe_audit_reports_known_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'summary.json').write_text(json.dumps([
                {'query':'membership','status':'received'}, {'query':'status','status':'received'},
                {'query':'dividend','status':'received'}]), encoding="utf-8")
            for date,codes in [('2024-01-01',['a','b']),('2025-01-01',['b','c'])]:
                (root/f'membership_{date}.json').write_text(json.dumps({'rows':[{'code':c} for c in codes]}), encoding="utf-8")
            report=audit_probe_directory(root)
            self.assertFalse(report['可发布正式包'])
            self.assertEqual(report['已验证能力']['两期样本各侧成分变化数'],1)
