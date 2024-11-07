from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from Trade import TradeInfo

@dataclass
class RegressionFormat:
    ticker: str
    industry: str
    y: list
    r: Decimal
    lv: Decimal
    om: Decimal
    date: datetime
    week_day: int
    sign_time: int

    def to_dict(self):
        return {
            'ticker': self.ticker,
            'industry': self.industry,
            **{f'y{i + 1}': Decimal(y_val) for i, y_val in enumerate(self.y)},
            'r': Decimal(self.r),
            'lv': Decimal(self.lv),
            'om': Decimal(self.om),
            'datetime': self.date,
            'week_day': int(self.week_day),
            'signal_time': int(self.sign_time)
        }
    
def tradeinfo_to_regression(trade_info: TradeInfo, y:list, r, om, week_day, signal_time, dt)->RegressionFormat:
    return RegressionFormat(
        ticker=trade_info.symbol,
        industry='',
        y=y,
        r=r,
        lv=Decimal("1"),
        om=om,
        date=dt,
        week_day=week_day,
        sign_time=signal_time
    )

@dataclass
class RegressionFormat2:
    ticker: str
    date: datetime
    week_day: int
    side: str
    open_y: list
    close_y: list
    signal_price: Decimal
    yesterday_close: Decimal
    today_open: Decimal
    sign_time: int
    threshold: Decimal
    open_amount: Decimal

    def to_dict(self):
        return {
            'ticker': self.ticker,
            'datetime': self.date,
            'week_day': int(self.week_day),
            'side': self.side,
            **{f'open price y{i + 1}': Decimal(y_val) for i, y_val in enumerate(self.open_y)},
            **{f'close price y{i + 1}': Decimal(y_val) for i, y_val in enumerate(self.close_y)},
            'signal price': Decimal(self.signal_price),
            'yesterday close': Decimal(self.yesterday_close),
            'today open': Decimal(self.today_open),
            'signal_time': int(self.sign_time),
            'open_amount': Decimal(self.open_amount),
            'threshold': Decimal(self.threshold),
        }
    
def tradeinfo_to_regression2(
        symbol:str, 
        dt: datetime, 
        week_day: int,
        side: str,
        open_y: list, 
        close_y: list, 
        signal_price: Decimal,
        yesterday_close: Decimal,
        today_open: Decimal,
        sign_time: int,
        open_amount: Decimal,
        threshold: Decimal
    )->RegressionFormat:
    return RegressionFormat2(
        ticker=symbol,
        date=dt,
        week_day=week_day,
        side=side,
        open_y=open_y,
        close_y=close_y,
        signal_price=signal_price,
        yesterday_close=yesterday_close,
        today_open=today_open,
        sign_time=sign_time,
        threshold=threshold,
        open_amount=open_amount
    )