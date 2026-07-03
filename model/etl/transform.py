import pandas as pd


class Transform:
    
    def __init__(self, raw_content: pd.DataFrame):
        self.df = raw_content.copy()
        
    def run(self):
         return (self
            ._fix_person_emp_length_()
            ._fix_loan_int_rate_null_()
            ._fix_age_outlier_()
            ._result_()
        )
         
    def _result_(self):
        return self.df
    
    def _fix_age_outlier_(self):
        mask = (self.df['person_age'] > 100) & (self.df['person_emp_length'].notna())
        
        self.df.loc[mask, "person_age"] = (
            18 + self.df.loc[mask, "person_emp_length"]
        )
        
        return self

    def _fix_person_emp_length_(self):
        mask_null = self.df['person_emp_length'].isna()
        
        for index in self.df[mask_null].index:
            income = self.df.loc[index, 'person_income']
            
            similar = self.df[
                (self.df['person_income'].between(left=income - 1500, right=income + 1500)) &
                (self.df['person_emp_length'].notna())
            ]
            
            if len(similar) > 0:
                self.df.loc[index, 'person_emp_length'] = similar['person_emp_length'].median()
        
        global_median = self.df['person_emp_length'].median()
        self.df['person_emp_length'] = self.df['person_emp_length'].fillna(global_median)
        
        return self
    
    def _fix_loan_int_rate_null_(self):
        self.df['loan_int_rate'] = self.df['loan_int_rate'].fillna(self.df.groupby('loan_grade')['loan_int_rate'].transform('median'))
        
        return self
    