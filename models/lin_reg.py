import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class Linear_Regression():
    def __init__(self, fit_intercept=True, n_jobs=-1):
        self.data_init = False 
        self.clf_x = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)
        self.clf_y = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)

    def init_data(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.data_init = True

    def fit(self):
        if self.data_init == False:
            raise ValueError("Data is not initialized")
        self.clf_x.fit(self.X_train, self.y_train[:, 0])
        self.clf_y.fit(self.X_train, self.y_train[:, 1])
        return self.clf_x, self.clf_y
    
    def predict(self):
        x_preds = self.clf_x.predict(self.X_test)
        y_preds = self.clf_y.predict(self.X_test)
        return x_preds, y_preds