from time import time

from optuna import create_study
from optuna.pruners import HyperbandPruner
from optuna.visualization import plot_optimization_history, plot_intermediate_values
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from tslearn.metrics import dtw
from frechetdist import frdist
import numpy as np

from models.random_forest_reg import Random_Forest_Regression
from models.xgboost import XGBoost


MODEL = None
X_TRAIN = None
Y_TRAIN = None
X_VAL = None
Y_VAL = None


def random_forest_opt(trial):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=10),
        # "criterion": trial.suggest_categorical("criterion", ['squared_error']),
        "max_depth": trial.suggest_int("max_depth", 2, 10, step=1),\
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 40, 66, 88]),
        # "ccp_alpha": trial.suggest_float("ccp_alpha", 0, 1, step=0.1),
        "random_state": 42,
        "warm_start": trial.suggest_categorical("warm_start", [True, False])
    }
    
    model = Random_Forest_Regression(**params)
    model.fit(X_TRAIN, Y_TRAIN)

    x, y = model.predict(X_VAL)
    preds = np.hstack((x[:, None], y[:, None]))

    return dtw(preds, Y_VAL)


def xgboost_opt(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=10),
        "learning_rate": trial.suggest_float('lr', 1e-6, 1),
        "max_depth": trial.suggest_int("max_depth", 2, 10, step=1),\
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'reg_lambda': trial.suggest_float('lambda', 1e-5, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-5, 1.0),
        'seed': 42
    }
    model = XGBoost(**params)
    model.fit(X_TRAIN, Y_TRAIN)

    x, y = model.predict(X_VAL)
    preds = np.hstack((x[:, None], y[:, None]))

    if np.isnan(preds).sum() > 0:
        return 20

    return dtw(preds, Y_VAL)


def optuna_optimize(model_name, iterations, X_train, y_train, X_val, y_val):
    global MODEL, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL
    X_TRAIN = X_train
    Y_TRAIN = y_train
    X_VAL = X_val
    Y_VAL = y_val

    study_name = f"model_name: {time():.0f}"

    study = create_study(
        load_if_exists=True,
        study_name=study_name,
        pruner=HyperbandPruner,
        direction="minimize",
    )

    match model_name:
        case "RF":
            study.optimize(random_forest_opt, n_trials=iterations)
        case "XG":
            study.optimize(xgboost_opt, n_trials=iterations)

    return study