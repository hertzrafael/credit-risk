from pathlib import Path

import pandas as pd
import os


class Load:
    
    def __init__(self, transformed_data: pd.DataFrame):
        self.df = transformed_data
        
        
    def run(self):
        self.df.to_csv(Path(os.getenv("TEMP_DIR")) / "credit_risk_data_gold.csv", index=False)
