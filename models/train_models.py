"""
Train classical ML models: Linear Regression and Random Forest.
Save models and print evaluation metrics.
"""
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def train_linear_regression(X_train, y_train, X_test, y_test, save_path):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test, save_path, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model, metrics
