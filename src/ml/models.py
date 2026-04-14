"""XGBoostAdapter: thin wrapper kept for save/load compatibility with scripts/predict.py."""
import os
import joblib


class XGBoostAdapter:
    """Wraps XGBClassifier with a uniform save/load interface."""

    def __init__(self, **params):
        self.params = params
        self.model  = None

    def fit(self, X, y, eval_set=None, verbose=False):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
