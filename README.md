# Recognition-of-Handwriting-from-Electromyography
This project is based on [Recognition-of-Handwriting-from-Electromyography](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006791). The original article used only a simple model (linear regression with Winner filter). We implemented more complex algorithms and assessed their quality.

## Results
Models we tried so far
 Linear Model |
 :---------|
![](examples/lin_reg_vis.png)|
 **XGBoost** |
![](examples/xgboost_vis.png)|
 **Random Forest** |
![](examples/rf_vis.png)|
 **KNN Regression** |
![](examples/knn_reg_vis.png)|
 **LSTM** |
![](examples/lstm_vis.png)|

Compare test data (orange) and prediction (blue)
 Linear Model |
 :---------|
![](examples/lr_trials_vis.png)|
 **XGBoost** |
![](examples/xg_trials_vis.png)|
 **Random Forest** |
![](examples/rf_trials_vis.png)|
 **KNN Regression** |
![](examples/knn_trials_vis.png)|
 **LSTM** |
![](examples/lstm_trials_vis.png)|