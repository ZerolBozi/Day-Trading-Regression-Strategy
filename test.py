import pandas as pd

def get_stocks_info(df: pd.DataFrame):
    public_company_datas = pd.read_csv('./datas/tw_datas/datas_filter_by_type/public.csv', encoding='utf-8-sig')
    otc_company_datas = pd.read_csv('./datas/tw_datas/datas_filter_by_type/otc.csv', encoding='utf-8-sig')

    public_company_datas = public_company_datas[['公司代號', '產業類別', '公司簡稱', '成立日期', '上市日期']]
    otc_company_datas = otc_company_datas[['公司代號', '產業類別', '公司簡稱', '成立日期', '上櫃日期']]

    public_company_codes = public_company_datas['公司代號'].values.tolist()
    otc_company_codes = otc_company_datas['公司代號'].values.tolist()

    df['name'] = df['ticker'].apply(company_name_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    df['industry'] = df['ticker'].apply(company_industry_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    df['found_date'] = df['ticker'].apply(company_found_date_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    df['appear_market_date'] = df['ticker'].apply(company_appear_date_apply_func, args=(public_company_datas, otc_company_datas, public_company_codes, otc_company_codes))
    cols = ['ticker', 'name', 'industry', 'found_date', 'appear_market_date', 'y1', 'y2', 'y3', 'y4', 'y5', 'r', 'lv', 'om','datetime', 'week_day', 'signal_time']
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
    ticker = int(ticker)
    if ticker in public_company_codes:
        return public_company_datas.loc[public_company_datas['公司代號'] == ticker, '上市日期'].iloc[0]
    elif ticker in otc_company_codes:
        return otc_company_datas.loc[otc_company_datas['公司代號'] == ticker, '上櫃日期'].iloc[0]
    
if __name__ == "__main__":
    df = pd.read_csv('./datas/tw_datas/trading_records/results12/results12.csv', encoding='utf-8-sig')
    new_df = get_stocks_info(df)
    new_df.to_csv('./datas/tw_datas/trading_records/results12/results_tw.csv', encoding='utf-8-sig', index=False)