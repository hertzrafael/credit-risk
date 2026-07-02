import pandas as pd


class Extract:
    
    def run(self) -> pd.DataFrame:
        return pd.read_csv('./files/credit_risk_dataset.csv')
    