import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Random_Forest_Regression():

    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        bootstrap=True,
        warm_start=False,
        max_features=1.0,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        random_state=0,
        n_jobs=-1,
    ):
        self.clf_x = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.clf_y = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            bootstrap=bootstrap,
            warm_start=warm_start,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train, y_train):
        self.clf_x.fit(X_train, y_train[:, 0])
        self.clf_y.fit(X_train, y_train[:, 1])
        return self.clf_x, self.clf_y 

    def predict(self, X_test):
        x_preds = self.clf_x.predict(X_test)
        y_preds = self.clf_y.predict(X_test)
        return x_preds, y_preds
