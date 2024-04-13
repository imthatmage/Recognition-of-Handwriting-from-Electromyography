import numpy as np
import xgboost as xgb

class XGBoost():
    def __init__(self, learning_rate=0.3, min_split_loss=0, max_depth=6, 
                 min_child_weight=1, n_estimators=100, sampling_method="uniform", 
                 reg_lambda=1, reg_alpha=1, objective="reg:squarederror", seed=0):
        
        self.clf_x = xgb.XGBRegressor(learning_rate=learning_rate, min_split_loss=min_split_loss, 
                                      max_depth=max_depth, min_child_weight=min_child_weight, 
                                      n_estimators=n_estimators, sampling_method=sampling_method, 
                                      reg_lambda=reg_lambda, reg_alpha=reg_alpha, objective=objective, seed=seed)
        
        self.clf_y = xgb.XGBRegressor(learning_rate=learning_rate, min_split_loss=min_split_loss, 
                                      max_depth=max_depth, min_child_weight=min_child_weight, 
                                      n_estimators=n_estimators, sampling_method=sampling_method, 
                                      reg_lambda=reg_lambda, reg_alpha=reg_alpha, objective=objective, seed=seed)

    def fit(self, X_train, y_train):
        self.clf_x.fit(X_train, y_train[:, 0])
        self.clf_y.fit(X_train, y_train[:, 1])
        return self.clf_x, self.clf_y 
    
    def predict(self, X_test):
        x_preds = self.clf_x.predict(X_test)
        y_preds = self.clf_y.predict(X_test)
        return x_preds, y_preds