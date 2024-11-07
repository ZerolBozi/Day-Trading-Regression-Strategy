import pandas as pd
import os

from Trade import TradeInfo
from Regression import RegressionFormat

class TradeRecord:
    def __init__(self, record_file_path: str, strategy_name: str):
        self.record_file_path = record_file_path
        self.strategy_name = strategy_name
        self.record_output_path = f"{self.record_file_path}/{self.strategy_name}"

        # check path exists
        os.mkdir(self.record_output_path) if not os.path.exists(self.record_output_path) else None

        # all trede records
        self.trade_records = {}
        self.regression_records = {}

    def add_regression_records(self, symbol: str, regression_info: RegressionFormat):
        if regression_info is None:
            return
        self.regression_records.setdefault(symbol, []).append(regression_info)

    def export_records(self):
        records = self.regression_records.copy()

        mode, header = ('a', False) if os.path.exists(f"{self.record_output_path}/{self.strategy_name}.csv") else ('w', True)
        dfs = []

        for _, data in records.items():
            df = pd.DataFrame([trade.to_dict() for trade in data])
            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(f"{self.record_output_path}/{self.strategy_name}.csv", mode=mode, header=header, index=False)