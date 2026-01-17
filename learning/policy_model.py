import numpy as np
from sklearn.linear_model import LogisticRegression

class LearnedSensingPolicy:
    """
    ML-based sensing policy using logistic regression.
    """

    def __init__(self):
        self.model = LogisticRegression()
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def select_action(self, features):
        """
        Predict whether high-quality sensing is needed.
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet")

        prob = self.model.predict_proba([features])[0][1]

        return "HIGH_QUALITY" if prob > 0.5 else "LOW_QUALITY"
