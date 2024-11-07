import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('./datas/us_datas/trading_records/results9/results9.csv', encoding='utf-8-sig')
    cols = ['ticker', 'industry', 'y1', 'y2', 'y3', 'y4', 'y5', 'r', 'lv', 'om', 'datetime', 'week_day', 'sign_time']
    df = df[cols]
    df.to_csv('./datas/us_datas/trading_records/results9/results.csv', encoding='utf-8-sig', index=False)