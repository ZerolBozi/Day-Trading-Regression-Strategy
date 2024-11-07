from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from Trade import TradeInfo

@dataclass
class StatisticsFormat:
    Ticker: str
    NPercentile: Decimal
    NDays: int
    TradePercentage: Decimal
    HoldDays: str
    OpenVolume: Decimal
    OpenAmount: Decimal
    VolumeThreshold: Decimal
    # OpenPrice: Decimal
    SignalTime: str
    OpenTradeTime: datetime
    CloseTradeTime: datetime
    OpenTradePrice: Decimal
    CloseTradePrice: Decimal
    OpenPriceDay: Decimal
    ClosePriceDay: Decimal
    Side: str
    GapThreshold: Decimal
    # 0 = NoGap, 1 = Gap+, -1 = Gap-
    IsGap: int
    # 0 = no limit, 15 = 15 minutes
    TimeLimit: int
    Profit: Decimal
    Return: Decimal

    def to_dict(self):
        return {
            'Ticker': self.Ticker.replace('.csv', ''),
            'NPercentile': Decimal(self.NPercentile),
            'NDays': int(self.NDays),
            'TradePercentage': Decimal(self.TradePercentage),
            'HoldDays': self.HoldDays,
            'OpenVolume': Decimal(self.OpenVolume),
            'OpenAmount': Decimal(self.OpenAmount),
            'VolumeThreshold': Decimal(self.VolumeThreshold),
            'SignalTime': self.SignalTime,
            'OpenTradeTime': self.OpenTradeTime,
            'CloseTradeTime': self.CloseTradeTime,
            'OpenTradePrice': Decimal(self.OpenTradePrice),
            'CloseTradePrice': Decimal(self.CloseTradePrice),
            'OpenPriceDay': Decimal(self.OpenPriceDay),
            'ClosePriceDay': Decimal(self.ClosePriceDay),
            'Side': self.Side,
            'GapThreshold': Decimal(self.GapThreshold),
            'IsGap': int(self.IsGap),
            'TimeLimit': int(self.TimeLimit),
            'Profit': Decimal(self.Profit),
            'Return': Decimal(self.Return)
        }

def TradeInfoToStatistics(trade_params, trade_info: TradeInfo, open_volume: Decimal, open_amount: Decimal, volume_threshold: int, signal_time: datetime, open_trade_time: datetime, open_trade_price: Decimal, open_price_day: Decimal, close_price_day: Decimal, gap_threshold: Decimal, is_gap: int)->StatisticsFormat:
    hold_days = str(trade_params.hold_days)
    if trade_params.end_of_week or trade_params.end_of_month:
        hold_days = "w" if trade_params.end_of_week else "m"

    side = "short" if trade_info.side == "buy" else "long"
    
    return StatisticsFormat(
        Ticker=trade_info.symbol,
        NPercentile=trade_params.n_percentile,
        NDays=trade_params.n_days,
        TradePercentage=trade_params.percentage,
        HoldDays=hold_days,
        OpenVolume=open_volume,
        OpenAmount=open_amount,
        VolumeThreshold=volume_threshold,
        SignalTime=signal_time,
        OpenTradeTime=open_trade_time,
        CloseTradeTime=trade_info.datetime,
        OpenTradePrice=open_trade_price,
        CloseTradePrice=trade_info.price,
        OpenPriceDay=open_price_day,
        ClosePriceDay=close_price_day,
        Side=side,
        GapThreshold=gap_threshold,
        IsGap=is_gap,
        TimeLimit=trade_params.time_param_hour*60 + trade_params.time_param_min,
        Profit=trade_info.income,
        Return=round((trade_info.income / trade_info.cost) * Decimal('100'), 4)
    )