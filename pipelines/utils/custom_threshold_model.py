from sklearn.base import BaseEstimator, ClassifierMixin

class CustomThresholdModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        # Pas besoin de ré-entraîner le modèle
        return self

    def predict(self, X):
        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)