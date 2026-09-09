"""人工构造的事件回放用例；不是市场数据或策略收益证据。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from astock_core.research.historical_inputs import ACTION_COLUMNS, load_historical_inputs, build_historical_features, sha256
from astock_core.research.oos_portfolio_research import create_execution_signal_panel, run_oos_portfolio_research, PROBABILITY_COLUMN
from astock_core.portfolio.portfolio_backtest import PortfolioConfig, run_portfolio_backtest
from astock_core.research.research_data import create_backtest_snapshot
from astock_core.research.research_summary import create_backtest_summary
from astock_core.research.strategy_backtest import run_strategy_backtest


def config(**kwargs):
    return PortfolioConfig(**dict(initial_capital=1000,max_positions=1,max_weight=1,lot_size=1,
                                  commission_rate=0,minimum_commission=0,sell_stamp_duty_rate=0,
                                  slippage_rate=0,rebalance_interval=5,**kwargs))


def panel(rows):
    return pd.DataFrame([{'可买入':True,'可卖出':True,'复权方式':'raw',**r} for r in rows])


def action(**kwargs):
    return {'事件编号':'e1','股票代码':'000001','登记日':'2025-01-01','除权日':'2025-01-02',
            '派息日':'2025-01-03','新增股份上市日':'2025-01-02','每股税后红利':1,
            '每股送转股数':0,'公告时间':'2024-12-30','类型':'distribution',**kwargs}


def write_bundle(root, mutate=None):
    directory=root/'data/research_execution'
    directory.mkdir(parents=True)
    dates=pd.bdate_range('2024-01-01',periods=100)
    members=pd.DataFrame({'日期':dates,'股票代码':['000001']*50+['000002']*50,'可得时间':dates+pd.Timedelta(hours=16)})
    prices=pd.DataFrame([{'日期':day,'股票代码':code,'收盘':10+np.sin(i)*.1+i*.01,'成交量':1000+i,
                         '复权方式':'raw','可买入':True,'可卖出':True,'可得时间':day+pd.Timedelta(hours=16),'状态来源':'合成测试状态'}
                        for code in ['000001','000002'] for i,day in enumerate(dates)])
    frames={'calendar':pd.DataFrame({'日期':dates}),'membership':members,'prices':prices,
            'corporate_actions':pd.DataFrame(columns=ACTION_COLUMNS)}
    if mutate:
        mutate(frames)
    manifest={'version':1,'universe_size':1,'corporate_actions_complete':True,'files':{}}
    for key,frame in frames.items():
        target=directory/f'{key}.csv'
        frame.to_csv(target,index=False)
        manifest['files'][key]={'path':target.name,'sha256':sha256(target),'source':'合成测试，不是实际历史'}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    return dates


class HistoricalInputTests(unittest.TestCase):
    def test_legacy_score_backtest_is_disabled_including_daily_snapshot(self):
        with self.assertRaisesRegex(ValueError, '旧评分回测已停用'):
            run_strategy_backtest()
        snapshot = create_backtest_snapshot()
        self.assertEqual(snapshot['状态'], 'rejected')
        self.assertEqual(snapshot['统计'], {})
        self.assertEqual(create_backtest_summary(snapshot)['状态'], 'rejected')

    def test_historical_membership_filters_signal_date_not_current_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dates=write_bundle(root)
            frames,audit=load_historical_inputs(root)
            features=build_historical_features(frames)
            self.assertEqual(set(features.loc[features['日期']<dates[50],'股票代码']),{'000001'})
            self.assertEqual(set(features.loc[features['日期']>=dates[50],'股票代码']),{'000002'})
            self.assertEqual(audit['状态'],'validated_inputs')
            self.assertFalse(audit['是否策略准入'])

    def test_future_known_constituents_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_bundle(root,lambda f:f['membership'].__setitem__('可得时间',f['membership']['日期']+pd.Timedelta(days=1)))
            with self.assertRaisesRegex(ValueError,'之后才可得'):
                load_historical_inputs(root)

    def test_cash_only_distribution_allows_blank_stock_listing_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def add_cash_event(frames):
                dates = frames['calendar']['日期']
                frames['corporate_actions'] = pd.DataFrame([
                    action(登记日=dates[20], 除权日=dates[21], 派息日=dates[22],
                           新增股份上市日='', 公告时间=dates[19], 每股送转股数=0)
                ])
            write_bundle(root, add_cash_event)
            frames, _ = load_historical_inputs(root)
            self.assertTrue(pd.isna(frames['corporate_actions'].iloc[0]['新增股份上市日']))

    def test_hash_changes_and_unknown_trade_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_bundle(root)
            p=root/'data/research_execution/prices.csv'
            p.write_text(p.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'校验值'):
                load_historical_inputs(root)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_bundle(root,lambda f:f['prices'].__setitem__('可买入',None))
            with self.assertRaisesRegex(ValueError,'不能假定可交易'):
                load_historical_inputs(root)

    def test_missing_historical_inputs_record_refusal_before_model_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch('astock_core.research.oos_portfolio_research.get_sklearn_dependencies') as sklearn:
            result=run_oos_portfolio_research(Path(tmp))
            sklearn.assert_not_called()
            self.assertEqual(result['status'],'insufficient')
            saved=json.loads(Path(result['metadata_file']).read_text())
            self.assertEqual(saved['状态'],'rejected')
            self.assertEqual(saved['组合统计'],{})

    def test_raw_execution_prices_used_in_integrated_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dates=write_bundle(root)
            market=root/'data/market';market.mkdir()
            pd.DataFrame({'日期':dates,'收盘':100+np.sin(np.arange(100)),'成交量':1000}).to_csv(market/'沪深300_sh000300.csv',index=False)
            evaluation={'ready':True,'windows':[],'out_of_sample':[pd.DataFrame({'日期':[dates[70]],'股票名称':['000002'],PROBABILITY_COLUMN:[.8]})]}
            with patch('astock_core.research.oos_portfolio_research.evaluate_rolling_windows',return_value=evaluation):
                result=run_oos_portfolio_research(root,config=config())
            self.assertEqual(result['status'],'success',result)
            trades=pd.read_csv(result['trades_file'])
            self.assertAlmostEqual(trades.iloc[0]['成交价格'],10+np.sin(71)*.1+71*.01,places=5)
            saved=json.loads(Path(result['metadata_file']).read_text())
            self.assertEqual(saved['审计']['状态'],'validated_inputs')


class ExecutionContractTests(unittest.TestCase):
    def test_future_price_gap_does_not_remove_past_signal(self):
        prices=panel([{'日期':'2025-01-01','股票名称':'A','收盘':10},
                      {'日期':'2025-01-02','股票名称':'A','收盘':10},
                      {'日期':'2025-01-03','股票名称':'B','收盘':10}])
        signals=[pd.DataFrame({'日期':['2025-01-01'],'股票名称':['A'],PROBABILITY_COLUMN:[.9]})]
        execution=create_execution_signal_panel(prices,signals,execution_prices=prices,trading_calendar=pd.date_range('2025-01-01',periods=3))
        self.assertEqual(execution.iloc[0]['股票'],'A')
        self.assertEqual(execution.attrs['因价格覆盖不足排除的股票'],[])
        with self.assertRaisesRegex(ValueError,'缺少持仓行情'):
            run_portfolio_backtest(execution,config())

    def test_late_listing_is_not_removed_for_missing_early_dates(self):
        prices=panel([{'日期':'2025-01-01','股票名称':'A','收盘':10},
                      {'日期':'2025-01-02','股票名称':'B','收盘':20},
                      {'日期':'2025-01-03','股票名称':'B','收盘':21}])
        signals=[pd.DataFrame({'日期':['2025-01-02'],'股票名称':['B'],PROBABILITY_COLUMN:[.9]})]
        execution=create_execution_signal_panel(prices,signals)
        self.assertEqual(execution.iloc[0]['股票'],'B')

    def test_missing_trade_permission_is_not_true(self):
        frame=pd.DataFrame([{'日期':'2025-01-01','股票':'A','收盘':10,'信号':1}])
        with self.assertRaisesRegex(ValueError,'交易状态'):
            run_portfolio_backtest(frame,config())
        frame=panel(frame.to_dict('records'));frame['可买入']=None
        with self.assertRaisesRegex(ValueError,'不能假定可交易'):
            run_portfolio_backtest(frame,config())

    def test_dividend_receivable_preserves_nav_before_cash_arrives(self):
        data=panel([{'日期':day,'股票':'000001','收盘':close,'信号':1} for day,close in [('2025-01-01',10),('2025-01-02',9),('2025-01-03',9)]])
        nav,_,stats=run_portfolio_backtest(data,config(),corporate_actions=[action()])
        self.assertEqual(nav['组合市值'].tolist(),[1000,1000,1000])
        self.assertEqual(nav['应收红利'].tolist(),[0,100,0])
        self.assertEqual(nav['现金'].tolist(),[0,0,100])
        self.assertEqual(stats['累计收益率'],0)

    def test_bonus_shares_locked_until_listing(self):
        data=panel([{'日期':day,'股票':code,'收盘':price,'信号':score}
                    for day in pd.date_range('2025-01-01',periods=4)
                    for code,price,score in [('000001',10 if day.day==1 else 5,1 if day.day==1 else 0),('000002',10,0 if day.day==1 else 1)]])
        cfg=PortfolioConfig(initial_capital=1000,max_positions=1,max_weight=1,rebalance_interval=1,lot_size=1,commission_rate=0,minimum_commission=0,sell_stamp_duty_rate=0,slippage_rate=0)
        nav,trades,_=run_portfolio_backtest(data,cfg,corporate_actions=[action(每股税后红利=0,每股送转股数=1,新增股份上市日='2025-01-04')])
        sells=trades[trades['方向']=='卖出']
        self.assertEqual(sells['日期'].tolist(),['2025-01-02','2025-01-04'])
        self.assertEqual(sells['股数'].tolist(),[100,100])
        self.assertTrue(nav['组合市值'].eq(1000).all())

    def test_initial_cost_included_and_missing_benchmark_not_filled(self):
        data=panel([{'日期':day,'股票':'A','收盘':10,'信号':1} for day in ['2025-01-01','2025-01-02']])
        cfg=PortfolioConfig(initial_capital=1000,max_positions=1,max_weight=1,lot_size=1,commission_rate=0,minimum_commission=5,sell_stamp_duty_rate=0,slippage_rate=0)
        _,_,stats=run_portfolio_backtest(data,cfg)
        self.assertAlmostEqual(stats['累计收益率'],-.005)
        self.assertAlmostEqual(stats['最大回撤'],-.005)
        with self.assertRaisesRegex(ValueError,'禁止前填'):
            run_portfolio_backtest(data,config(),pd.DataFrame({'日期':['2025-01-01'],'收盘':[100]}))

    def test_suspended_holding_does_not_allow_extra_position(self):
        data=panel([{'日期':'2025-01-01','股票':'A','收盘':10,'信号':1},
                    {'日期':'2025-01-01','股票':'B','收盘':10,'信号':0},
                    {'日期':'2025-01-02','股票':'A','收盘':10,'信号':0,'可卖出':False},
                    {'日期':'2025-01-02','股票':'B','收盘':10,'信号':1}])
        cfg=PortfolioConfig(initial_capital=1000,max_positions=1,max_weight=.5,rebalance_interval=1,lot_size=1,commission_rate=0,minimum_commission=0,sell_stamp_duty_rate=0,slippage_rate=0)
        nav,trades,stats=run_portfolio_backtest(data,cfg)
        self.assertEqual(len(trades),1)
        self.assertEqual(nav.iloc[-1]['持仓数量'],1)
        self.assertEqual(stats['未成交记录'][0]['原因'],'显式禁止卖出')

class CausalityTests(unittest.TestCase):
    def test_future_distribution_does_not_rewrite_past_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dates=write_bundle(root)
            frames,_=load_historical_inputs(root)
            before=build_historical_features(frames)
            changed={k:v.copy() for k,v in frames.items()}
            changed['corporate_actions']=pd.DataFrame([action(股票代码='000002',登记日=dates[79],除权日=dates[80],派息日=dates[80],新增股份上市日=dates[80],公告时间=dates[78],每股税后红利=1)])
            after=build_historical_features(changed)
            from astock_core.research.prediction_features import FEATURE_COLUMNS
            mask=before['日期']<dates[75]
            pd.testing.assert_frame_equal(before.loc[mask,FEATURE_COLUMNS].reset_index(drop=True),after.loc[mask,FEATURE_COLUMNS].reset_index(drop=True))

    def test_five_day_label_does_not_jump_over_missing_market_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dates=write_bundle(root)
            frames,_=load_historical_inputs(root)
            frames['prices']=frames['prices'][~((frames['prices']['日期']==dates[30]) & (frames['prices']['股票代码']=='000001'))]
            features=build_historical_features(frames)
            row=features[(features['日期']==dates[26]) & (features['股票代码']=='000001')].iloc[0]
            self.assertTrue(pd.isna(row['future_5d_return']))

    def test_missing_member_price_refuses_experiment_instead_of_dropping_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def mutate(frames):
                frames['prices']=frames['prices'].iloc[1:]
            write_bundle(root,mutate)
            with self.assertRaisesRegex(ValueError,'拒绝整个实验'):
                load_historical_inputs(root)

    def test_adjusted_execution_prices_rejected(self):
        data=panel([{'日期':'2025-01-01','股票':'A','收盘':10,'信号':1,'复权方式':'qfq'}])
        with self.assertRaisesRegex(ValueError,'前复权价格成交'):
            run_portfolio_backtest(data,config())

    def test_buy_blocked_does_not_replace_ranking_with_future_tradable_stock(self):
        data=panel([{'日期':'2025-01-01','股票':'A','收盘':10,'信号':1,'可买入':False},
                    {'日期':'2025-01-01','股票':'B','收盘':10,'信号':.5}])
        nav,trades,stats=run_portfolio_backtest(data,config())
        self.assertTrue(trades.empty)
        self.assertEqual(nav.iloc[0]['现金'],1000)
        self.assertEqual(stats['未成交记录'][0]['股票'],'A')
