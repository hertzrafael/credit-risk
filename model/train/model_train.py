from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer

from config import MODELS

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os


class ModelTrain:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self):
        y = self.data['loan_status']
        X = self.data.drop('loan_status', axis=1)
        
        # OBTENDO VARIAVEIS CATEGORIAS E QUANTITATIVAS
        quant_columns = X.select_dtypes(include=["number"]).columns.tolist()
        cat_columns = [
            column for column in X.columns
            if X[column].dtype == "object" and column != "loan_grade"
        ]
        
        print(f"COLUNAS QUANTITATIVAS: {quant_columns}")
        print(f"COLUNAS CATEGÓRICAS: {cat_columns}")
        
        # SEPARAÇÃO DOS DADOS EM TREINO E TESTE
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
        
        results = []
        for model, model_config in MODELS.items():
            print(f"Iniciando modelagem de {model}")
            
            with mlflow.start_run(run_name=model, ):
                transformer_methods = [
                    ("num", StandardScaler() if model_config["linear"] else "passthrough", quant_columns),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_columns),
                    ("grade", OrdinalEncoder(categories=[["A", "B", "C", "D", "E", "F", "G"]]), ["loan_grade"])
                ]
                
                preprocess = ColumnTransformer(transformers=transformer_methods)
                pipeline_methods = [
                    ("preprocess", preprocess),
                    ("model", model_config['model'])
                ]
                
                pipeline = Pipeline(pipeline_methods)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
                
                grid = GridSearchCV(pipeline, model_config['params'], scoring='roc_auc', cv=cv)
                grid.fit(X_train, y_train)
                
                model = grid.best_estimator_
                y_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores[:-1])
                best_threshold = thresholds[best_idx]
                y_pred = (y_proba >= best_threshold).astype(int)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                # ML FLOW LOGGING
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                
                mlflow.log_metric("best_threshold", best_threshold)
                
                # SALVANDO MODELO NO MLFLOW
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    name="model",
                    serialization_format="pickle"
                )
                
                results.append({
                    "Name": model,
                    "Estimator": grid.best_estimator_,
                    "Best Score": grid.best_score_,
                    "Best Threshold": best_threshold
                })
        
        results_df = pd.DataFrame(results).sort_values("Best Score", ascending=False)
        
        best_row = results_df.iloc[0]
        best_model = best_row["Estimator"]
        best_threshold = best_row["Best Threshold"]
        
        joblib.dump({"model": best_model, "threshold": best_threshold}, os.getenv("MODEL_PATH"))
