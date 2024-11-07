import pandas as pd
import numpy as np
from decimal import Decimal

def quantile_exc(ser: pd.Series, q: Decimal):
    """
    Calculate the exclusive quantile of a pandas series.

    :param ser: A pandas series.
    :type ser: pandas.Series

    :param q: The quantile value (between 0 and 1).
    :type q: Decimal

    :return: The quantile value.

    ### This function is based on the solution provided at: https://stackoverflow.com/questions/38596100/python-equivalent-of-excels-percentile-exc
    """
    ser_sorted = ser.sort_values()
    rank = q * (len(ser) + 1) - 1
    assert rank > 0, 'Quantile is too small'
    rank_l = rank.quantize(Decimal('1.'), rounding='ROUND_DOWN')
    return ser_sorted.iat[int(rank)] + (ser_sorted.iat[int(rank) + 1] - ser_sorted.iat[int(rank)]) * float((rank - rank_l))

def entry_indicator(datas: list|pd.Series, percentile: Decimal=Decimal('0.95'), n_days: int = 90, percentile_method: str = 'normal') -> Decimal:
    """
    Calculate the entry indicator.

    :param datas: A list of data.
    :type datas: list | pandas.Series

    :param percentile: The percentile value (between 0 and 1).
    :type percentile: Decimal
    :default percentile: 0.95

    :param n_days: The number of days.
    :type n_days: int
    :default n_days: 90

    :param percentile_method: The method to calculate the percentile value.
    :type percentile_method: str
    :default percentile_method: 'normal'
    :available percentile_method: 'normal', 'lower', 'linear', 'higher'

    :return: The entry indicator value.
    """
    if len(datas) != n_days:
        raise Exception('The length of datas should be equal to n_days.')
    
    if percentile > 1 or percentile < 0:
        raise Exception('The value of percentile should be between 0 and 1.')
    
    if percentile_method not in ['normal', 'lower', 'linear', 'higher']:
        raise Exception('The value of percentile_method should be one of the following: normal, lower, linear, higher.')

    series_data = pd.Series(datas) if isinstance(datas, list) else datas

    if percentile_method == 'normal':
        percentile_value = quantile_exc(series_data, percentile)
    else:
        percentile_value = np.percentile(series_data, percentile * 100, method=percentile_method)

    return percentile_value

def trade_signal_tmp(market_open_price: Decimal, now_price: Decimal, percentage: Decimal=Decimal('0.03')):
    """
    Calculate the trade signal.

    :param market_open_price: The market open price.
    :type market_open_price: Decimal

    :param now_price: The now price.
    :type now_price: Decimal

    :param percentage: The percentage value (between 0 and 1).
    :type percentage: Decimal
    :default percentage: 0.03
    """
    upper_limit = market_open_price * (1 + percentage)
    lower_limit = market_open_price * (1 - percentage)

    if now_price > upper_limit:
        # sell
        return -1
    if now_price < lower_limit:
        # buy
        return 1
    
    return 0
    
if __name__ == '__main__':
    datas = pd.read_csv('TSLA.US.csv')
    open_volumes_group = datas.groupby('Date')['Volume'].first()
    open_volumes = open_volumes_group[-90:].values.tolist()
    print(entry_indicator(open_volumes))