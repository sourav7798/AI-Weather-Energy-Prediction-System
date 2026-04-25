"""
Forecasting module: predict solar and wind energy generation (kW) using
trained models and simple physical proxies.
"""
import os
import joblib
import numpy as np
import pandas as pd

def predict_with_model(model_path, X):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    model = joblib.load(model_path)
    preds = model.predict(X)
    return preds

def solar_power_prediction(df_features, model_path=None, panel_capacity_kw=0.3, panel_count=10):
    """Estimate solar power (kW). If model_path provided, use model.
    Otherwise use simple proxy: irradiance * panel area efficiency.
    """
    if model_path:
        preds = predict_with_model(model_path, df_features)
        return preds
    # Simple physical proxy: solar_irradiance in W/m2 -> kW per panel approx
    # Assume panel area 1.6 m2 and efficiency 0.18
    area = 1.6
    eff = 0.18
    irradiance = df_features.get('solar_irradiance')
    if isinstance(irradiance, (pd.Series, np.ndarray)):
        power_per_panel_w = irradiance * area * eff
        total_kw = power_per_panel_w * panel_count / 1000.0
        return total_kw
    else:
        return None

def wind_power_prediction(df_features, model_path=None, turbine_rated_kw=200):
    """Estimate wind power (kW). If model_path provided, use model.
    Otherwise use cubic wind-speed proxy scaled to rated.
    """
    if model_path:
        preds = predict_with_model(model_path, df_features)
        return preds
    ws = df_features.get('wind_speed')
    # cubic proxy normalized to rated at 12 m/s
    def proxy(v):
        return turbine_rated_kw * min((v / 12.0) ** 3, 1.0)

    if isinstance(ws, (pd.Series, np.ndarray)):
        return ws.apply(proxy) if isinstance(ws, pd.Series) else np.array([proxy(v) for v in ws])
    else:
        return None
