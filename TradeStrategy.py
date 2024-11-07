import os
import tqdm
import pandas as pd
import datetime
import calendar
import itertools
import multiprocessing
from pydantic import BaseModel
from functools import partial
from decimal import Decimal

from Record import TradeRecord
from Trade import FakeTrade
from Indicators import entry_indicator
from Regression import tradeinfo_to_regression, tradeinfo_to_regression2

class TradeParams(BaseModel):
    market_type: str
    strategy_name: str = ""
    n_percentile: Decimal = Decimal('0.95')
    n_days: int = 90
    # trade_mode ['once', 'hedge', 'one_way', 'increase']
    trade_mode: str = 'one_way'
    percentage: Decimal = Decimal('0.05')
    trade_size: Decimal = Decimal('1')
    trade_value: int = 0
    use_stop_loss: bool = False
    stop_loss_value: Decimal = Decimal('0.05')
    keep_trade: bool = False
    use_time_param: bool = False
    time_param_hour: int = 0
    time_param_min: int = 0
    use_trade_interval: bool = False
    trade_interval_hour: int = 0
    trade_interval_min: int = 0
    use_avg_price: bool = False
    hold_days: int = 0
    end_of_week: bool = False
    week_param: int = 0
    end_of_month: bool = False
    month_param: int = 0
    gap_threshold: Decimal = Decimal('0.03')
    use_amount: bool = False
    use_close_position_time: bool = False
    close_position_hour: int = 0
    close_position_min: int = 0
    reversed_strategy: bool = False
    # reversed_condition
    record_path: str = ""
    is_public: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self.record_path = f'./datas/{self.market_type}_datas/trading_records'
        if self.strategy_name == '':
            self.strategy_name = f'{self.trade_mode}_{self.n_percentile}_{self.n_days}_{self.percentage}_{self.trade_size if self.trade_size > 0 else self.trade_value}_{self.stop_loss_value}_[{self.time_param_hour},{self.time_param_min}]_{"avg" if self.use_avg_price else ""}_{self.hold_days}_{self.end_of_week}_{self.week_param}_{self.end_of_month}_{self.month_param}'

def get_stock_datas(stock_path:str, stock_filter:list=[])->dict:
    stock_list = os.listdir(stock_path)
    stock_datas = dict()
    for stock in stock_list:
        stock = stock.replace('.csv', '')
        if stock_filter == [] or stock in stock_filter:
            stock_datas.setdefault(stock, f'{stock_path}/{stock}.csv')
    return stock_datas

class TradeStrategy:
    def __init__(self, trade_param:TradeParams)->None:
        self.trade_param = trade_param
        self.signal_datas = dict()

    def start_strategy(self, symbol:str, source_path:str, trade_record:TradeRecord):
        self.trade_record = trade_record
        source_data = pd.read_csv(source_path)
        self.create_trading_signal(symbol, source_data, use_amount=self.trade_param.use_amount)
        self.simulate_trading(symbol, self.signal_datas[symbol])

    def get_trade_record(self):
        return self.trade_record
    
    def create_trading_signal(self, symbol:str, source_data:pd.DataFrame, percentile_method:str = 'normal', use_amount:bool = False, save_csv:bool = False, save_path:str = './trading_signal.csv'):
        """
        Create the trading signal for a given date based on volume percentiles.

        :param source_data: A pandas DataFrame containing the trading data.
        :type source_data: pandas.DataFrame

        :param percentile_method: The method used for calculating the percentile.
        :type percentile_method: str
        :method list: [normal, lower(numpy), linear(numpy), higher(numpy)]
        :default: 'normal'

        :param save_csv: Whether to save the results in CSV format.
        :type save_csv: bool
        :default: True

        :param save_path: The file path to save the results in CSV format.
        :type save_path: str
        :default: './trading_signal.csv'
        """
        signal_data = source_data[source_data['IsOpen'] == 1].copy() if 'IsOpen' in source_data.columns else source_data.copy()
        signal_data['Date'] = pd.to_datetime(signal_data['Date'], format='%Y-%m-%d').dt.date
        signal_data['Threshold'] = 0
        signal_data['Signal'] = -1

        signal_data['Amount'] = signal_data['Volume'] * signal_data['Close']
        
        signal_colume = 'Volume'

        if use_amount:
            signal_colume = 'Amount'

        open_volumes_group = signal_data.groupby('Date')[signal_colume].first()
        unique_dates = signal_data['Date'].unique()[self.trade_param.n_days:]

        for date in unique_dates:
            open_volumes = open_volumes_group[open_volumes_group.index < date][-self.trade_param.n_days:]
            percentile_value = entry_indicator(open_volumes, self.trade_param.n_percentile, self.trade_param.n_days, percentile_method)
            open_volume = signal_data[signal_data['Date'] == date][signal_colume].values[0]
            signal_data.loc[signal_data['Date'] == date, 'Threshold'] = percentile_value
            signal_data.loc[signal_data['Date'] == date, 'Signal'] = 1 if open_volume > percentile_value else 0

        if save_csv:
            signal_data.to_csv(save_path, index=False)

        self.signal_datas.setdefault(symbol, signal_data)

    def get_close_dates_by_diff(self, diff:int, sign:datetime.date, source_unique_dates:list)->list:
        close_dates = []
        for d in range(0, diff + 1):
            _sign = sign + datetime.timedelta(days=d)
            if _sign in source_unique_dates:
                close_dates.append(_sign)
        return close_dates

    def get_close_date(self, signal_date:datetime.date, source_unique_dates:list)->datetime.date:
        sign = signal_date

        if self.trade_param.end_of_week:  
            diff = calendar.FRIDAY - sign.weekday()
            return self.get_close_dates_by_diff(diff, sign, source_unique_dates)

        if self.trade_param.end_of_month:
            diff = calendar.monthrange(signal_date.year, signal_date.month)[1] - signal_date.day
            return self.get_close_dates_by_diff(diff, sign, source_unique_dates)

        if self.trade_param.hold_days >= 0:
            close_dates = []
            d = 0
            while len(close_dates) != self.trade_param.hold_days:
                _sign = sign + datetime.timedelta(days=d)
                d += 1
                if _sign in source_unique_dates:
                    close_dates.append(_sign)
                if _sign > source_unique_dates[-1]:
                    break

            return close_dates
    
    def simulate_trading(self, symbol:str, source_data:pd.DataFrame):
        self.trade = FakeTrade()

        # astype
        source_data['Open'] = source_data['Open'].apply(str)
        source_data['High'] =  source_data['High'].apply(str)
        source_data['Low'] = source_data['Low'].apply(str)
        source_data['Close'] = source_data['Close'].apply(str)
        source_data['Volume'] = source_data['Volume'].apply(str)
        source_data['Amount'] = source_data['Amount'].apply(str)
        source_data['Threshold'] = source_data['Threshold'].apply(str)

        source_data['Open'] = source_data['Open'].apply(Decimal)
        source_data['High'] =  source_data['High'].apply(Decimal)
        source_data['Low'] = source_data['Low'].apply(Decimal)
        source_data['Close'] = source_data['Close'].apply(Decimal)
        source_data['Volume'] = source_data['Volume'].apply(Decimal)
        source_data['Amount'] = source_data['Amount'].apply(Decimal)
        source_data['Threshold'] = source_data['Threshold'].apply(Decimal)

        source_data['DateTime'] = pd.to_datetime(source_data['DateTime']).dt.to_pydatetime()
        source_data['Date'] = pd.to_datetime(source_data['Date'], format='%Y-%m-%d').dt.date
        source_data['Time'] = pd.to_datetime(source_data['Time'], format='%H:%M:%S').dt.time

        # get the signal data
        signal_data = source_data[source_data['Signal'] == 1].copy()

        # 分為原始資料的日期與訊號資料的日期, 原始資料日期用於平倉, end_of_week, end_of_month, hold_days會使用到
        source_unique_dates = list(source_data['Date'].unique())
        signal_unique_dates = list(signal_data['Date'].unique())

        for signal_date in signal_unique_dates:
            # initialize the trade variables
            open_positions = {'sell': None, 'buy': None}
            
            # 為了計算是否gap, 所以需要取得前一天的收盤價
            previous_date = source_unique_dates[source_unique_dates.index(signal_date) - 1]
            previous_data = source_data[source_data['Date'] == previous_date]
            previous_close_price = Decimal(previous_data['Close'].values[-1])

            # get the signal data
            source_trade_datas = signal_data[signal_data['Date'] == signal_date]
            # pass the first row, because it is market open
            trade_datas = source_trade_datas[1:] if len(source_trade_datas) > 1 else source_trade_datas

            # 取得開盤時間, 當日開盤價, gap值
            open_datetime = pd.to_datetime(source_trade_datas['DateTime'].values[0])
            open_price_day = source_trade_datas['Open'].values[0]
            open_amount = source_trade_datas['Amount'].values[0]
            threshold = source_trade_datas['Threshold'].values[0]
            gaps = (open_price_day - previous_close_price) / previous_close_price
            
            # 計算進入條件: 開盤價 * (1 - 下跌百分比) or 開盤價 * (1 + 上漲百分比)
            entry_buy_price = open_price_day * (1 - self.trade_param.percentage)
            entry_sell_price = open_price_day * (1 + self.trade_param.percentage)

            # 使用時間限制, 限制在開盤後的幾分鐘內做交易, 超過則不做交易
            if self.trade_param.use_time_param:
                time_param = datetime.timedelta(hours=self.trade_param.time_param_hour, minutes=self.trade_param.time_param_min)
                open_time = source_trade_datas['Time'].values[0]
                time_param_check = (datetime.datetime.combine(signal_date, open_time) + time_param).time()

            # 連續交易才會用到的 (increase)
            if self.trade_param.use_trade_interval:
                trade_interval = datetime.timedelta(hours=self.trade_param.trade_interval_hour, minutes=self.trade_param.trade_interval_min)
                last_trade_time = None
                trade_interval_time = None

            # 主循環, 循環當天的每一根k棒並判斷進入條件
            for _ , trade_data in trade_datas.iterrows():
                
                # get open position
                long_position = open_positions.get('buy', None)
                short_position = open_positions.get('sell', None)

                # 取得交易需要的變數, 交易時間, 最低價, 最高價, 開倉方向, 停損價格, 是否需要停損
                trade_datetime = trade_data['DateTime']
                low_price = trade_data['Low']
                high_price = trade_data['High']
                open_position_side = long_position.side if long_position is not None else short_position.side if short_position is not None else None
                stop_loss_size = open_positions.get(open_position_side, None)
                is_need_to_stop = False if not self.trade_param.use_stop_loss else open_positions[open_position_side].stopPrice[stop_loss_size.size] > low_price if open_position_side == 'buy' else open_positions[open_position_side].stopPrice[stop_loss_size.size] < high_price

                # 是否需要停損
                if is_need_to_stop:
                    # close the position
                    close_position_info = self.trade.close_order(
                        symbol=symbol,
                        order_id=open_positions[open_position_side].orderId,
                        price=open_positions[open_position_side].stopPrice[stop_loss_size.size],
                        size=open_positions[open_position_side].size,
                        datetime=trade_datetime,
                        is_stop=True
                    )
                    # add the trade record
                    self.trade_record.add_trade_record(symbol, close_position_info)
                    self.trade_record.add_income_record(symbol, close_position_info)
                    self.trade_param.trade_value += close_position_info.totalIncome
                    # reset the open position variable
                    open_positions[open_position_side] = None
                    # 如果持續交易的話, 則繼續循環, 否則break主循環
                    if self.trade_param.keep_trade:
                        last_trade_time = trade_data['Time']
                        trade_interval_time = (datetime.datetime.combine(signal_date, last_trade_time) + trade_interval).time()
                        continue
                    break
                
                # 是否使用限制時間
                if self.trade_param.use_time_param:
                    # 超過時間, 停止交易, break主循環
                    if trade_data['Time'] > time_param_check:
                        break
                
                # 是否使用交易間隔
                if self.trade_param.use_trade_interval:
                    # 這裡邏輯有點錯誤, 因為沒有使用到, 所以暫時不管它
                    if last_trade_time is not None:
                        if trade_data['Time'] < trade_interval_time:
                            break
                
                # 這是為了之後使用模擬金所設定的判斷, 但一樣沒有使用到, 所以暫時不管它
                if self.trade_param.trade_size == 0:
                    if self.trade_param.trade_value < 0:
                        break
                            
                # 判斷進入條件並 Set order parameters, 否則跳過此次循環
                if low_price < entry_buy_price:
                    order_price = low_price if not self.trade_param.use_avg_price else (low_price + high_price) / 2
                    # if the price is new low, it should close the short position first
                    close_open_position = short_position
                    open_position = long_position
                    order_side = 'buy'
                    stop_loss_price = order_price * (1 - self.trade_param.stop_loss_value) if self.trade_param.use_stop_loss else None
                elif high_price > entry_sell_price:
                    order_price = high_price if not self.trade_param.use_avg_price else (low_price + high_price) / 2
                    # if the price is new high, it should close the long position first
                    close_open_position = long_position
                    open_position = short_position
                    order_side = 'sell'
                    stop_loss_price = order_price * (1 + self.trade_param.stop_loss_value) if self.trade_param.use_stop_loss else None
                else:
                    continue
                
                # 單一方向
                if self.trade_param.trade_mode == 'one_way':
                    if close_open_position is not None:
                        close_position_info, open_trade_price, open_trade_time = self.trade.close_order(
                            symbol=symbol,
                            order_id=close_open_position.orderId,
                            price=order_price,
                            size=close_open_position.size,
                            datetime=trade_datetime
                        )
                        
                        # 計算信號到結算的漲跌幅
                        y = (order_price - open_trade_price) / open_trade_price
                        # 計算開盤到信號的漲跌幅
                        r = (open_price_day - open_trade_price) / open_trade_price
                        # 昨日收盤到今天開盤的漲跌幅 gaps
                        time_difference = open_trade_time - open_datetime
                        signal_time = str(time_difference).replace('0 days ', '')
                        interval = time_difference.total_seconds() / 3600
                        interval = int(interval) // 1
                        weekday = trade_datetime.weekday() + 1

                        regression_info = tradeinfo_to_regression(close_position_info, [y], r, gaps,weekday, interval, signal_date)
                        self.trade_record.add_regression_records(symbol, regression_info)
                        open_positions[close_open_position.side] = None

                # 看參數是指定size(股數)還是value(金額), 如果是value, 則需要計算size, 也是為了之後使用模擬金所設定的判斷, 但一樣沒有使用到, 所以暫時不管它
                trade_size = self.trade_param.trade_size if self.trade_param.trade_size > 0 else self.trade_param.trade_value // order_price

                # 開單
                if open_position is None:
                    # open a new position
                    open_position_info = self.trade.open_order(
                        symbol=symbol,
                        side=order_side,
                        price=order_price,
                        size=trade_size,
                        datetime=trade_datetime + datetime.timedelta(seconds=60) if self.trade_param.trade_mode == 'one_way' else trade_datetime,
                        stop_price=stop_loss_price
                    )
                    open_positions[order_side] = open_position_info
                else:
                    # increase the position
                    if self.trade_param.trade_mode == 'increase':
                        open_position_info = self.trade.increase_position(
                            symbol=symbol,
                            order_id=open_position.orderId,
                            price=order_price,
                            size=trade_size,
                            datetime=trade_datetime,
                            stop_price=stop_loss_price
                        )
                        open_positions[order_side] = open_position_info
                
                # once是只交易一次 所以直接break主循環
                if self.trade_param.trade_mode == 'once':
                    break
            
            # 主循環已結束, 進行平倉
            # get close date, 根據不同的平倉參數, 取得平倉日期
            if open_positions != {'sell': None, 'buy': None}:
                close_dates = self.get_close_date(signal_date, source_unique_dates)
                close_datas = [source_data[source_data['Date'] == close_date] for close_date in close_dates]

                open_prices = []
                close_prices = []
                # 根據平倉日期取得當日的資料
                if self.trade_param.use_close_position_time:
                    close_position_time = datetime.time(hour=self.trade_param.close_position_hour, minute=self.trade_param.close_position_min)
                    minute = self.trade_param.close_position_min
                    for close_data in close_datas:
                        while close_position_time not in close_data['Time'].values:
                            minute += 1
                            if minute > 59:
                                close_position_time = close_data['Time'].tail(1).values[0]
                                break
                            close_position_time = datetime.time(hour=close_position_time.hour, minute=minute)
                        close_data = close_data[close_data['Time'] == close_position_time]
                        close_prices.append(close_data['Close'].values[0])
                else:
                    for close_data in close_datas:
                        open_data = close_data.head(1)
                        open_prices.append(open_data['Open'].values[0])

                        _close_data = close_data.tail(1)
                        close_prices.append(_close_data['Close'].values[0])

            # 將所有開倉的單子進行平倉
            for _ , position_info in open_positions.items():
                if position_info is not None:
                    close_position_info, open_trade_price, open_trade_time = self.trade.close_order(
                        symbol=symbol,
                        order_id=position_info.orderId,
                        price=close_prices[-1],
                        size=position_info.size,
                        datetime=close_data['DateTime'].values[0]
                    )
                    y = []
                    for idx in range(len(close_prices)):
                        if idx == 0:
                            y.append((close_prices[idx] - open_trade_price) / open_trade_price)
                        else:
                            y.append((close_prices[idx] - close_prices[idx - 1]) / close_prices[idx - 1])
                                      
                    # 計算開盤到信號的漲跌幅
                    r = (open_trade_price - open_price_day) / open_price_day
                    # 昨日收盤到今天開盤的漲跌幅 gaps
                    time_difference = open_trade_time - open_datetime
                    interval = time_difference.total_seconds() / 3600
                    interval = int(interval) // 1
                    weekday = trade_datetime.weekday() + 1

                    regression_info = tradeinfo_to_regression(close_position_info, y, r, gaps,weekday, interval, signal_date)
                    # regression_info = tradeinfo_to_regression2(
                    #     close_position_info.symbol,
                    #     signal_date,
                    #     weekday,
                    #     position_info.side,
                    #     open_prices,
                    #     close_prices,
                    #     open_trade_price,
                    #     previous_close_price,
                    #     open_price_day,
                    #     interval,
                    #     open_amount,
                    #     threshold
                    # )
                    self.trade_record.add_regression_records(symbol, regression_info)

def start_event(source_datas: dict, trade_param: TradeParams, num_processes: int):
    s = TradeStrategy(trade_param)
    
    start_event_func = partial(s.start_strategy)

    multiprocessing.Manager().register('TradeRecord', TradeRecord)

    manager = multiprocessing.Manager()
    trade_record = manager.TradeRecord(trade_param.record_path, trade_param.strategy_name)
    
    with tqdm.tqdm(total=len(list(source_datas.keys())), desc='Simulate Trading') as pbar:
        pool = multiprocessing.Pool(num_processes)
        async_results = []

        for symbol in source_datas.keys():
            result = pool.apply_async(start_event_func, args=(symbol, source_datas[symbol], trade_record), callback=lambda _: pbar.update(1))
            async_results.append(result)
        
        for result in async_results:
            result.get()

        pool.close()
        print('Trade Params:', trade_param.n_percentile, trade_param.n_days, trade_param.percentage, trade_param.gap_threshold, 'Done!')
        with open('log.txt', 'a+') as f:
            f.write(f'{trade_param.n_percentile}, {trade_param.n_days}, {trade_param.percentage}, {trade_param.gap_threshold}\n')

    trade_record.export_records()

if __name__ == "__main__":
    market_type = 'tw'
    stock_path = f'./datas/{market_type}_datas/history_datas'
    # stock_filter = pd.read_csv('./datas/tw_datas/group_by_volume/big_volumes.csv')['0'].values.tolist()
    stock_filter = ['1236']
    source_datas = get_stock_datas(stock_path, stock_filter)

    n_percentile: Decimal = [Decimal('0.95')]
    n_days: int = [120]
    trade_mode: str = ['once']
    percentage: Decimal = [Decimal('0.05')]
    use_time_param: bool = [False]
    time_param_hour: int = [0]
    time_param_min: int = [0]
    hold_days: int = [5]
    end_of_week: bool = [False]
    end_of_month: bool = [False]
    gap_threshold: Decimal = [Decimal('0.03')]
    use_amount = True
    num_processes = multiprocessing.cpu_count() - 2 if stock_filter == [] else len(stock_filter)
    # num_processes = 2
    use_close_position_time = False
    close_position_hour = 0
    close_position_min = 0
    # num_processes = 1

    parameter_combinations = list(itertools.product(n_percentile, n_days, trade_mode, percentage, use_time_param, time_param_hour, time_param_min, hold_days, end_of_week, end_of_month, gap_threshold))

    for param_combination in parameter_combinations:
        n_percentile, n_days, trade_mode, percentage, use_time_param, time_param_hour, time_param_min, hold_days, end_of_week, end_of_month, gap_threshold = param_combination
        # strategy_name = f'{n_days}_{n_percentile}_strategy_{"end_of_week" if end_of_week else "end_of_month" if end_of_month else f"day_trading" if hold_days == 0 else "hold_days_1"}_{percentage}_limit_time_{time_param_hour}_{time_param_min}'
        strategy_name = 'results13'
        trade_param = TradeParams(market_type=market_type, strategy_name=strategy_name, n_percentile=n_percentile, n_days=n_days, trade_mode=trade_mode, percentage=percentage, use_time_param=use_time_param, time_param_hour=time_param_hour, time_param_min=time_param_min, hold_days=hold_days, end_of_week=end_of_week, end_of_month=end_of_month, gap_threshold=gap_threshold, use_amount=use_amount, is_public=False, use_close_position_time=use_close_position_time, close_position_hour=close_position_hour, close_position_min=close_position_min)
        start_event(source_datas, trade_param, num_processes)