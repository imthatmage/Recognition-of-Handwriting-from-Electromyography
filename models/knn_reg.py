import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class KNN_Regression():
    def __init__(self, x_n_neighbors=5, y_n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, 
                 p=2, metric='minkowski', metric_params=None, n_jobs=-1):
        
        self.clf_x = KNeighborsRegressor(n_neighbors=x_n_neighbors, weights=weights, 
                                         algorithm=algorithm, leaf_size=leaf_size, p=p, 
                                         metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        
        self.clf_y = KNeighborsRegressor(n_neighbors=y_n_neighbors, weights=weights, 
                                         algorithm=algorithm, leaf_size=leaf_size, p=p, 
                                         metric=metric, metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, X_train, y_train):
        self.clf_x.fit(X_train, y_train[:, 0])
        self.clf_y.fit(X_train, y_train[:, 1])
        return self.clf_x, self.clf_y 
    
    def predict(self, X_test):
        x_preds = self.clf_x.predict(X_test)
        y_preds = self.clf_y.predict(X_test)
        return x_preds, y_preds