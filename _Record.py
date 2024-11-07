import pandas as pd
import os
import json

from Trade import TradeInfo
from Statistics import StatisticsFormat

class TradeRecord:
    def __init__(self, record_path:str, strategy_name:str):
        self.record_path = record_path
        self.strategy_name = strategy_name
        self.trade_output_path = f"{record_path}/{strategy_name}/trading-records"
        self.income_output_path = f"{record_path}/{strategy_name}/income-records"
        self.statistics_output_path = f"{record_path}/{strategy_name}/statistics-records"
        self.log_path = f"{record_path}/{strategy_name}/logs"
        self.json_output_path = f"{record_path}/{strategy_name}/json-records"
        self.trades = {}
        self.statistics = {}
        self.incomes = {}
        # self.datas = {}

    def check_path_exists(self):
        os.makedirs(self.json_output_path) if not os.path.exists(self.json_output_path) else None
        os.makedirs(self.trade_output_path) if not os.path.exists(self.trade_output_path) else None
        os.makedirs(self.income_output_path) if not os.path.exists(self.income_output_path) else None
        os.makedirs(self.statistics_output_path) if not os.path.exists(self.statistics_output_path) else None

    def reset(self):
        self.trades = {}
        self.incomes = {}
        self.statistics = {}
        self.datas = {}

    def get_trade_records(self, symbol: str=None):
        return self.trades.get(symbol, self.trades)

    def get_income_records(self, symbol: str=None):
        return self.incomes.get(symbol, self.incomes)

    def add_trade_record(self, symbol: str, trade_info: TradeInfo, statistics_info: StatisticsFormat):
        if trade_info is None:
            return
        self.trades.setdefault(symbol, []).append(trade_info)

        if statistics_info is not None:
            self.statistics.setdefault(symbol, []).append(statistics_info)

    def add_income_record(self, symbol: str, trade_info: TradeInfo):
        if trade_info is None:
            return
        self.incomes.setdefault(symbol, []).append(trade_info)

    def set_output_path(self, record_path:str, strategy_name:str):
        self.record_path = record_path
        self.strategy_name = strategy_name
        self.trade_output_path = f"{record_path}/{strategy_name}/trading-records"
        self.income_output_path = f"{record_path}/{strategy_name}/income-records"
        self.statistics_output_path = f"{record_path}/{strategy_name}/statistics-records"
        self.log_path = f"{record_path}/{strategy_name}/logs"
        self.json_output_path = f"{record_path}/{strategy_name}/json-records"

    def export_to_csv(self):
        self.check_path_exists()
        trade_datas = self.trades.copy()

        for symbol, trade_data in trade_datas.items():
            df = pd.DataFrame([trade.__dict__ for trade in trade_data])
            cols = ['datetime', 'createTime'] + [col for col in df.columns if col not in ['datetime', 'createTime']]
            df = df[cols]
            df.to_csv(f"{self.trade_output_path}/{symbol}-records.csv", index=False)
            df.to_json(f"{self.trade_output_path}/{symbol}-records.json", orient='records', date_format='iso', default_handler=str)

            # source_dataframe = self.datas.get(symbol)
            # source_dataframe['DateTime'] = pd.to_datetime(source_dataframe['DateTime'])
            # df['datetime'] = pd.to_datetime(df['datetime'])
            # merged_df = source_dataframe.merge(df, left_on='DateTime', right_on='datetime', how='left')
            
            # selected_columns = ['Timestamp', 'DateTime', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            #                     'orderType', 'size', 'price', 'side', 'income', 'incomeRate', 'totalIncome', 'totalIncomeRate']
            
            # merged_df = merged_df[selected_columns]
            # merged_df['orderType'].fillna('N/A', inplace=True)
            # merged_df['size'].fillna(0, inplace=True)
            # merged_df['price'].fillna(0.0, inplace=True)
            # merged_df['side'].fillna('N/A', inplace=True)
            # merged_df['income'].fillna(0.0, inplace=True)
            # merged_df['incomeRate'].fillna(0.0, inplace=True)
            # merged_df['totalIncome'].fillna(0.0, inplace=True)
            # merged_df['totalIncomeRate'].fillna(0.0, inplace=True)

            # merged_df.to_csv(f"{self.trade_output_path}/{symbol}-trading-records.csv", index=False)
            # merged_df.to_json(f"{self.trade_output_path}/{symbol}-trading-records.json", orient='records', date_format='iso', default_handler=str)

        income_datas = self.incomes.copy()

        for symbol, income_data in income_datas.items():
            df = pd.DataFrame([income.__dict__ for income in income_data])
            cols = ['datetime', 'createTime'] + [col for col in df.columns if col not in ['datetime', 'createTime']]
            df = df[cols]
            df.to_csv(f"{self.income_output_path}/{symbol}-incomes.csv", index=False)
            df.to_json(f"{self.income_output_path}/{symbol}-incomes.json", orient='records', date_format='iso', default_handler=str)

        statistics_datas = self.statistics.copy()

        for symbol, statistics_data in statistics_datas.items():
            df = pd.DataFrame([statistics.__dict__ for statistics in statistics_data])
            df['DateTime'] = df['Date'].copy()
            df['Date'] = df['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))
            cols = ['Ticker', 'Date', 'OpenVolume', 'VolumeThreshold', 'OpenPrice', 'ClosePrice', 'Price', 'Side', 'Profit']
            df = df[cols]
            df.to_csv(f"{self.statistics_output_path}/{symbol}-statistics.csv", index=False)
            df.to_json(f"{self.statistics_output_path}/{symbol}-statistics.json", orient='records', date_format='iso', default_handler=str)

    def export_statistics(self):
        os.makedirs(f"{self.record_path}/{self.strategy_name}") if not os.path.exists(f"{self.record_path}/{self.strategy_name}") else None
        statistics_datas = self.statistics.copy()

        if os.path.exists(f"{self.record_path}/{self.strategy_name}/{self.strategy_name}.csv"):
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True

        amount_path = self.record_path.replace(self.record_path.split('/')[-1],'')

        big_amount_list = [symbol.replace('.csv', '') for symbol in pd.read_csv(f'{amount_path}/group_by_amount/big_amount.csv')['0'].values.tolist()]
        mid_amount_list = [symbol.replace('.csv', '') for symbol in pd.read_csv(f'{amount_path}/group_by_amount/mid_amount.csv')['0'].values.tolist()]
        small_amount_list = [symbol.replace('.csv', '') for symbol in pd.read_csv(f'{amount_path}/group_by_amount/small_amount.csv')['0'].values.tolist()]

        for _, statistics_data in statistics_datas.items():
            df = pd.DataFrame([statistics.to_dict() for statistics in statistics_data])
            df = get_stocks_info(df)
            df['GroupByAmount'] = df['Ticker'].apply(lambda x: 1 if x in small_amount_list else 2 if x in mid_amount_list else 3 if x in big_amount_list else 0)
            df.to_csv(f"{self.record_path}/{self.strategy_name}/{self.strategy_name}.csv", header=header, index=False, mode=mode, encoding='utf-8')
            mode = 'a'
            header = False

    def export_to_json(self, symbol: str = None):
        self.check_path_exists()
        if symbol is None:
            json_output_path = f"{self.json_output_path}/"
        else:
            json_output_path = f"{self.json_output_path}/{symbol}-"

        trade_data = self.trades.get(symbol, self.trades)
        with open(f"{json_output_path}records.json", 'w') as f:
            json.dump(trade_data, f, default=lambda o: o.to_dict())

        income_data = self.incomes.get(symbol, self.incomes)
        with open(f"{json_output_path}income.json", 'w') as f:
            json.dump(income_data, f, default=lambda o: o.to_dict())

        statistics_data = self.statistics.get(symbol, self.statistics)
        with open(f"{json_output_path}statistics.json", 'w') as f:
            json.dump(statistics_data, f, default=lambda o: o.to_dict())

    def export_logs(self):
        self.check_path_exists()
        income_datas = self.incomes.copy()
        calculate_results = {}
        paper_results = {}

        for symbol, trades in income_datas.items():
            trade_count = len(trades)
            win_count = 0
            trade_date = {}
            trade_date_income = {}
            positive_profit = []
            negative_profit = []
            win_rate = 0
            total_profit = 0
            average_profit_per_trading = 0
            average_profit = 0
            average_loss = 0
            max_drawdown = 0
            max_drawdown_date = ''
            odds = 0
            expectation_of_net_profit = 0

            #1104
            _data = {
                'total_buy':0, 'buy_win': 0, 'buy_loss': 0, 'buy_profits': 0.0,
                'total_sell':0, 'sell_win': 0, 'sell_loss': 0, 'sell_profits': 0.0,
                'buy_cost': 0.0, 'sell_cost': 0.0, 'total_cost':0.0
            }

            for trade in trades:
                side = 'buy' if trade.side == "sell" else 'sell'
                if trade.totalIncome > 0:
                    win_count += 1
                    positive_profit.append(trade.totalIncome)
                    _data[f"{side}_win"] += 1
                elif trade.totalIncome < 0:
                    negative_profit.append(trade.totalIncome)
                    _data[f"{side}_loss"] += 1

                _data[f"total_{side}"] += 1
                _data[f"{side}_cost"] += trade.total
                _data['total_cost'] += trade.total
                _data[f"{side}_profits"] += trade.totalIncome

                trade_date.setdefault(
                    trade.datetime.strftime("%Y-%m-%d %H:%M:%S"), 
                    {
                        'totalIncome': trade.totalIncome,
                        'totalIncomeRate': trade.totalIncomeRate,
                    }
                )

                trade_date_income.setdefault(
                    trade.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    trade.totalIncome
                )

            # datas/logs/{trade_count}/{symbol}.txt
            # ex: datas/logs/1/A.US.txt
            log_path = f"{self.log_path}/{trade_count}"
            os.makedirs(log_path) if not os.path.exists(log_path) else None

            win_rate = win_count / trade_count
            total_profit = sum(trade_date_income.values())
            average_profit_per_trading = total_profit / trade_count
            positive_profit_sum = sum(positive_profit)
            negative_profit_sum = sum(negative_profit)
            average_profit = positive_profit_sum / len(positive_profit) if len(positive_profit) > 0 else 0
            average_loss = negative_profit_sum / len(negative_profit) if len(negative_profit) > 0 else 0
            max_drawdown = min(trade_date_income.values()) if len(negative_profit) > 0 else 0
            max_drawdown_date = min(trade_date_income, key=trade_date_income.get) if max_drawdown != 0 else ''
            # 不確定是否需要絕對值，論文內沒有絕對值
            if positive_profit_sum > 0 and abs(negative_profit_sum) > 0:
                odds = positive_profit_sum / abs(negative_profit_sum)
            # 這公式很奇怪，論文給出的公式是這樣的
            expectation_of_net_profit = win_rate * (1 + odds) - 1

            with open(f"{log_path}/{symbol}.txt", 'w') as f:
                f.write(f"Total Trading Count: {trade_count}\n")
                f.write(f"Win Count: {win_count}\n")
                f.write(f"Win Rate: {win_rate}\n")
                f.write(f"Total Profit: {total_profit}\n")
                f.write(f"Average Profit Per Trading: {average_profit_per_trading}\n")
                f.write(f"Average Profit: {average_profit}\n")
                f.write(f"Average Loss: {average_loss}\n")
                f.write(f"Maximum Drawdown: {max_drawdown}\n")
                f.write(f"Maximum Drawdown Date: {max_drawdown_date}\n")
                f.write(f"Odds: {odds}\n")
                f.write(f"Expectation of net profit: {expectation_of_net_profit}\n")
                f.write(f"Trading Date:\n")
                for date, income_dict in trade_date.items():
                    f.write(f"{date}: {income_dict.get('totalIncome', 0)}, {income_dict.get('totalIncomeRate', 0)}\n")

            symbol_result = {
                'total_trading': trade_count,
                'win': win_count,
                'win_rate': win_rate,
                'total_profits': total_profit,
                'average_profit_per_trading': average_profit_per_trading,
                'average_profit': average_profit,
                'average_loss': average_loss,
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_drawdown_date,
                'odds': odds,
                'expectation_of_net_profit': expectation_of_net_profit,
            }

            paper_result = {
                'total_trading': trade_count,
                'win': win_count,
                'win_rate': win_rate,
                'total_cost': _data['total_cost'],
                'total_profits': total_profit,
                'total_return': total_profit / _data['total_cost'] * 100 if _data['total_cost'] != 0 else 0,
                'long_trading': _data['total_buy'],
                'long_win': _data['buy_win'],
                'long_loss': _data['buy_loss'],
                'long_win_rate': _data['buy_win'] / _data['total_buy'] if _data['total_buy'] != 0 else 0,
                'long_cost': _data['buy_cost'],
                'long_profits': _data['buy_profits'],
                'long_return': _data['buy_profits'] / _data['buy_cost'] * 100 if _data['buy_cost'] != 0 else 0,
                'short_trading': _data['total_sell'],
                'short_win': _data['sell_win'],
                'short_loss': _data['sell_loss'],
                'short_win_rate': _data['sell_win'] / _data['total_sell'] if _data['total_sell'] != 0 else 0,
                'short_cost': _data['sell_cost'],
                'short_profits': _data['sell_profits'],
                'short_return': _data['sell_profits'] / _data['sell_cost'] * 100 if _data['sell_cost'] != 0 else 0,
            }

            calculate_results.setdefault(symbol, symbol_result)
            paper_results.setdefault(symbol, paper_result)

            # output json for website
            # with open(f"{self.log_path}/{symbol}-log.json", 'w') as f:
            #     json.dump(symbol_result, f, indent=4)
        
        with open(f"{self.log_path}/results_{self.strategy_name}.json", 'w') as f:
            json.dump(calculate_results, f, indent=4)

        with open(f"{self.json_output_path}/logs.json", 'w') as f:
            json.dump(calculate_results, f, indent=4)

        csv_rows = []

        for symbol, data in calculate_results.items():
            csv_row = [symbol]
            csv_row.extend(data.values())
            csv_rows.append(csv_row)

        columns = ["symbol", "total_trading", "win", "win_rate", "total_profits", "average_profit_per_trading", "average_profit", "average_loss", "max_drawdown", "max_drawdown_date", "odds", "expectation_of_net_profit"]
        df = pd.DataFrame(csv_rows, columns=columns)

        df_sorted = df.sort_values(by="symbol")

        df_sorted.to_csv(f"{self.log_path}/results_{self.strategy_name}.csv", index=False)

        csv_rows = []

        for symbol, data in paper_results.items():
            csv_row = [symbol]
            csv_row.extend(data.values())
            csv_rows.append(csv_row)

        columns = ["symbol", "total_trading", "win", "win_rate", "total_cost", "total_profits", "total_return", "long_trading", "long_win", "long_loss", "long_win_rate", "long_cost", "long_profits", "long_return", "short_trading", "short_win", "short_loss", "short_win_rate", "short_cost", "short_profits", "short_return"]
        df = pd.DataFrame(csv_rows, columns=columns)

        df_sorted = df.sort_values(by="symbol")

        df_sorted.to_csv(f"{self.log_path}/results_{self.strategy_name}_for_paper.csv", index=False)

    def export_record(self):
        income_datas = self.incomes.copy()
        total_count = 0
        _data = {
            'buy_win': 0, 'sell_win': 0, 'total_win': 0, 'buy_loss': 0, 'sell_loss': 0, 'total_loss': 0, 
            'buy_profits': 0.0, 'sell_profits': 0.0, 'total_profits': 0.0, 'buy_ret': 0.0, 'sell_ret': 0.0, 'total_ret': 0.0,
            'buy_cost': 0.0, 'sell_cost': 0.0, 'total_cost':0.0
        }
        for _, trades in income_datas.items():

            for trade in trades:
                side = 'buy' if trade.side == "sell" else 'sell'
                if trade.totalIncome > 0:
                    _data[f"{side}_win"] += 1
                    _data['total_win'] += 1
                else:
                    _data[f"{side}_loss"] += 1
                    _data['total_loss'] += 1
                
                _data[f"{side}_cost"] += trade.total
                _data['total_cost'] += trade.total
                _data[f"{side}_profits"] += trade.totalIncome
                _data['total_profits'] += trade.totalIncome

        total_buy = _data['buy_win'] + _data['buy_loss']
        total_sell = _data['sell_win'] + _data['sell_loss']
        total_count = total_buy + total_sell

        buy_win_rate = round(_data['buy_win'] / total_buy * 100 ,2)
        sell_win_rate = round(_data['sell_win'] / total_sell * 100 ,2)
        total_win_rate = round(_data['total_win'] / total_count * 100 ,2)

        buy_return = round(_data['buy_profits'] / _data['buy_cost'] * 100, 3)
        sell_return = round(_data['sell_profits'] / _data['sell_cost'] * 100, 3)
        total_return = round(_data['total_profits'] / _data['total_cost'] * 100, 3)

        with open(f"{self.record_path}/{self.strategy_name}/{self.strategy_name}.txt", 'w') as f:
            f.write(f"Buy Win: {_data['buy_win']}\n")
            f.write(f"Buy Loss: {_data['buy_loss']}\n")
            f.write(f"Buy Win Rate: {buy_win_rate}%\n")
            f.write(f"Buy Return: {buy_return}%\n")

            f.write(f"Sell Win: {_data['sell_win']}\n")
            f.write(f"Sell Loss: {_data['sell_loss']}\n")
            f.write(f"Sell Win Rate: {sell_win_rate}%\n")
            f.write(f"Sell Return: {sell_return}%\n")

            f.write(f"Total Win: {_data['total_win']}\n")
            f.write(f"Total Loss: {_data['total_loss']}\n")
            f.write(f"Total Win Rate: {total_win_rate}%\n")
            f.write(f"Total Return: {total_return}%\n")

def get_stocks_info(df: pd.DataFrame):
    public_company_datas = pd.read_csv('./datas/tw_datas/datas_filter_by_type/public.csv', encoding='utf-8-sig')
    otc_company_datas = pd.read_csv('./datas/tw_datas/datas_filter_by_type/otc.csv', encoding='utf-8-sig')

    public_company_datas = public_company_datas[['公司代號', '產業類別', '公司簡稱', '成立日期', '上市日期']]
    otc_company_datas = otc_company_datas[['公司代號', '產業類別', '公司簡稱', '成立日期', '上櫃日期']]

    public_company_codes = public_company_datas['公司代號'].values.tolist()
    otc_company_codes = otc_company_datas['公司代號'].values.tolist()

    df['CompanyName'] = df['Ticker'].apply(company_name_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    df['CompanyIndustry'] = df['Ticker'].apply(company_industry_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    df['CompanyFoundDate'] = df['Ticker'].apply(company_found_date_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    df['CompanyAppearMarketDate'] = df['Ticker'].apply(company_appear_date_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    
    cols = ['Ticker', 'CompanyName', 'CompanyIndustry', 'CompanyFoundDate', 'CompanyAppearMarketDate','NPercentile', 'NDays', 'TradePercentage', 'HoldDays', 'OpenVolume', 'OpenAmount', 'VolumeThreshold', 'GapThreshold', 'OpenTradeTime', 'CloseTradeTime', 'SignalTime', 'OpenTradePrice', 'CloseTradePrice', 'OpenPriceDay', 'ClosePriceDay', 'Side', 'IsGap', 'TimeLimit', 'Profit', 'Return']
    df = df[cols]

    return df

def company_name_apply_func(ticker, public_company_datas, otc_company_datas, public_company_codes, otc_company_codes):
    ticker = int(ticker)
    if ticker in public_company_codes:
        return public_company_datas.loc[public_company_datas['公司代號'] == ticker, '公司簡稱'].iloc[0]
    elif ticker in otc_company_codes:
        return otc_company_datas.loc[otc_company_datas['公司代號'] == ticker, '公司簡稱'].iloc[0]
    
def company_industry_apply_func(ticker, public_company_datas, otc_company_datas, public_company_codes, otc_company_codes):
    ticker = int(ticker)
    if ticker in public_company_codes:
        return public_company_datas.loc[public_company_datas['公司代號'] == ticker, '產業類別'].iloc[0]
    elif ticker in otc_company_codes:
        return otc_company_datas.loc[otc_company_datas['公司代號'] == ticker, '產業類別'].iloc[0]
    
def company_found_date_apply_func(ticker, public_company_datas, otc_company_datas, public_company_codes, otc_company_codes):
    ticker = int(ticker)
    if ticker in public_company_codes:
        return public_company_datas.loc[public_company_datas['公司代號'] == ticker, '成立日期'].iloc[0]
    elif ticker in otc_company_codes:
        return otc_company_datas.loc[otc_company_datas['公司代號'] == ticker, '成立日期'].iloc[0]
    
def company_appear_date_apply_func(ticker, public_company_datas, otc_company_datas, public_company_codes, otc_company_codes):
    if ticker in public_company_codes:
        return public_company_datas.loc[public_company_datas['公司代號'] == ticker, '上市日期'].iloc[0]
    elif ticker in otc_company_codes:
        return otc_company_datas.loc[otc_company_datas['公司代號'] == ticker, '上櫃日期'].iloc[0]