from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


MODELS = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=12),
        "linear": True,
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=12),
        "linear": False,
        "params": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [5, 10, None],
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=12, eval_metric='logloss'),
        "linear": False,
        "params": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 6],
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=12),
        "linear": False,
        "params": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__num_leaves": [31, 63],
            "model__max_depth": [-1, 10, 20],
        }
    }
}