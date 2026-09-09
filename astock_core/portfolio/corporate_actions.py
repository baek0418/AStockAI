"""显式权益分派台账：应收现金与新增股份分别按实际日期处理。"""
import math
import pandas as pd


class CorporateActionLedger:
    def __init__(self, actions=None):
        self.events = [] if actions is None else pd.DataFrame(actions).to_dict('records')
        self.entitlements = {}
        self.receivables = {}
        self.locks = []
        self.journal = []
        self.blocked_orders = []
        for event in self.events:
            if event.get('类型') != 'distribution':
                raise ValueError('未支持的公司行为；禁止用复权价代替现金和股份处理。')
            for column in ('登记日','除权日'):
                event[column] = pd.Timestamp(event[column])
            for column in ('每股税后红利','每股送转股数'):
                if not math.isfinite(float(event[column])) or float(event[column]) < 0:
                    raise ValueError('公司行为金额/比例无效。')
            cash_event = float(event['每股税后红利']) > 0
            stock_event = float(event['每股送转股数']) > 0
            if cash_event and pd.isna(event.get('派息日')):
                raise ValueError('现金分红缺少派息日。')
            if stock_event and pd.isna(event.get('新增股份上市日')):
                raise ValueError('送转股缺少新增股份上市日。')
            event['派息日'] = pd.Timestamp(event['派息日']) if cash_event else event['除权日']
            event['新增股份上市日'] = pd.Timestamp(event['新增股份上市日']) if stock_event else event['除权日']
            if event['登记日'] >= event['除权日'] or event['派息日'] < event['除权日'] or event['新增股份上市日'] < event['除权日']:
                raise ValueError('公司行为日期顺序无效。')
        ids = [event['事件编号'] for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError('公司行为事件编号重复。')

    def start_day(self, day, positions, cash):
        day = pd.Timestamp(day)
        for event in self.events:
            event_id, symbol = event['事件编号'], str(event['股票代码'])
            if event['除权日'] == day:
                entitled = self.entitlements.get(event_id, 0)
                extra = entitled * float(event['每股送转股数'])
                if abs(extra-round(extra)) > 1e-8:
                    raise ValueError('送转产生零碎股，缺少零碎股处置依据，拒绝继续。')
                if extra:
                    positions[symbol] = positions.get(symbol,0) + int(round(extra))
                    self.locks.append((symbol,int(round(extra)),event['新增股份上市日']))
                amount = entitled * float(event['每股税后红利'])
                self.receivables[event_id] = amount
                if entitled:
                    self.journal.append({'日期':str(day.date()),'事件编号':event_id,'股票':symbol,
                                         '登记股数':entitled,'新增股数':int(round(extra)),'应收红利':amount})
            if event['派息日'] <= day and event_id in self.receivables:
                amount = self.receivables.pop(event_id)
                cash += amount
                if amount:
                    self.journal.append({'日期':str(day.date()),'事件编号':event_id,'股票':symbol,'到账现金':amount})
        self.locks = [item for item in self.locks if item[2] > day]
        return cash

    def end_day(self, day, positions):
        for event in self.events:
            if event['登记日'] == pd.Timestamp(day):
                self.entitlements[event['事件编号']] = positions.get(str(event['股票代码']),0)

    def locked(self, symbol):
        return sum(shares for code,shares,_ in self.locks if code == symbol)

    @property
    def receivable_value(self):
        return sum(self.receivables.values())
