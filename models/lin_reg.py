import numpy as np
from sklearn.linear_model import LinearRegression

class Linear_Regression():
    def __init__(self, fit_intercept=True, n_jobs=-1):
        self.clf = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        return self.clf
    
    def predict(self, X_test):
        preds = self.clf.predict(X_test)
        return preds[:, 0], preds[:, 1]