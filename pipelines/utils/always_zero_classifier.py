import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class AlwaysZeroClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)
    
    def predict_proba(self, X):
        return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])

