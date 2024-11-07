import pandas as pd

df = pd.read_csv('./datas/tw_datas/trading_records/results1/results1.csv')
# print(df)
unique_tickers = df['ticker'].unique().tolist()

data = pd.DataFrame()
data['ticker'] = unique_tickers

public_company_datas = pd.read_csv('./datas/tw_datas/datas_filter_by_type/public.csv', encoding='utf-8-sig')
otc_company_datas = pd.read_csv('./datas/tw_datas/datas_filter_by_type/otc.csv', encoding='utf-8-sig')

public_company_datas = public_company_datas[['公司代號', '產業類別', '公司簡稱', '成立日期', '上市日期']]
otc_company_datas = otc_company_datas[['公司代號', '產業類別', '公司簡稱', '成立日期', '上櫃日期']]

public_company_codes = public_company_datas['公司代號'].values.tolist()
otc_company_codes = otc_company_datas['公司代號'].values.tolist()

def company_name_apply_func(ticker, public_company_datas, otc_company_datas, public_company_codes, otc_company_codes):
    ticker = int(ticker)
    if ticker in public_company_codes:
        return public_company_datas.loc[public_company_datas['公司代號'] == ticker, '公司簡稱'].iloc[0]
    elif ticker in otc_company_codes:
        return otc_company_datas.loc[otc_company_datas['公司代號'] == ticker, '公司簡稱'].iloc[0]
    
data['name'] = data['ticker'].apply(company_name_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
data.to_csv('codes.csv', index=False)