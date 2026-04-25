"""
Flask dashboard exposing routes for visualization and simple controls.
Uses Plotly for interactive charts embedded in HTML.
Includes live weather data, AI predictions, and grid optimization.
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from api.data_collection import load_sample
from forecasting.forecasting import solar_power_prediction, wind_power_prediction
from optimization.grid_optimization import balance_supply_demand, distribute_load, recommend_battery_size
import threading, time, os
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# ── Live data store (updated by background thread) ──────────────────
live_weather_history = []
LIVE_LOCK = threading.Lock()

def _simulate_live():
    """Background thread: generate realistic weather ticks every 5 seconds."""
    np.random.seed(int(time.time()) % 10000)
    while True:
        t = datetime.now()
        hour_f = t.hour + t.minute / 60.0
        temp = 22 + 8 * np.sin((hour_f - 6) * np.pi / 12) + np.random.normal(0, 0.5)
        humidity = max(30, min(95, 65 - 15 * np.sin((hour_f - 6) * np.pi / 12) + np.random.normal(0, 3)))
        wind = max(0, 4 + 3 * np.sin(hour_f * np.pi / 6) + np.random.normal(0, 1))
        pressure = 1013 + np.random.normal(0, 1)
        clouds = max(0, min(100, 30 + 20 * np.cos(hour_f * np.pi / 8) + np.random.normal(0, 8)))
        if 5 <= hour_f <= 19:
            irr = max(0, 800 * np.sin((hour_f - 5) * np.pi / 14) * (1 - clouds / 200) + np.random.normal(0, 20))
        else:
            irr = 0
        rec = {
            'datetime': t.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind, 1),
            'pressure': round(pressure, 1),
            'clouds': round(clouds, 1),
            'solar_irradiance': round(irr, 1)
        }
        with LIVE_LOCK:
            live_weather_history.append(rec)
            if len(live_weather_history) > 200:
                live_weather_history.pop(0)
        time.sleep(5)

_thread = threading.Thread(target=_simulate_live, daemon=True)
_thread.start()

# ── API: live data endpoint ──────────────────────────────────────────
@app.route('/api/live')
def api_live():
    with LIVE_LOCK:
        data = list(live_weather_history)
    if not data:
        return jsonify({'data': [], 'grid': {}, 'total_supply': 0, 'total_demand': 0})
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    solar = solar_power_prediction(df, model_path=None)
    wind = wind_power_prediction(df, model_path=None)
    supply_arr = solar + wind
    np.random.seed(42)
    demand_arr = 400 + 100 * np.sin(np.linspace(0, 3.14, len(df))) + np.random.normal(0, 10, len(df))
    grid = balance_supply_demand(
        supply_arr.values if hasattr(supply_arr, 'values') else np.array(supply_arr),
        demand_arr
    )
    result = []
    for i, row in enumerate(data):
        r = dict(row)
        r['solar_kw'] = round(float(solar.iloc[i]) if hasattr(solar, 'iloc') else float(solar[i]), 2)
        r['wind_kw'] = round(float(wind.iloc[i]) if hasattr(wind, 'iloc') else float(wind[i]), 2)
        r['supply_kw'] = round(float(supply_arr.iloc[i]) if hasattr(supply_arr, 'iloc') else float(supply_arr[i]), 2)
        r['demand_kw'] = round(float(demand_arr[i]), 2)
        result.append(r)
    return jsonify({
        'data': result,
        'grid': grid,
        'total_supply': round(float(np.sum(supply_arr)), 2),
        'total_demand': round(float(np.sum(demand_arr)), 2)
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to accept a JSON payload of features and return predictions."""
    payload = request.json
    df = pd.DataFrame(payload)
    solar = solar_power_prediction(df, model_path=None)
    wind = wind_power_prediction(df, model_path=None)
    return jsonify({'solar_kw': solar.tolist() if hasattr(solar, 'tolist') else None,
                    'wind_kw': wind.tolist() if hasattr(wind, 'tolist') else None})

# ── Main page ────────────────────────────────────────────────────────
@app.route('/')
def index():
    df = load_sample('data/sample_data.csv')
    solar_kw = solar_power_prediction(df, model_path=None)
    wind_kw = wind_power_prediction(df, model_path=None)
    np.random.seed(42)
    demand_kw = 400 + 100 * np.sin(np.linspace(0, 3.14, len(df))) + np.random.normal(0, 20, len(df))
    supply = solar_kw + wind_kw
    grid_status = balance_supply_demand(
        supply.values if hasattr(supply, 'values') else supply,
        demand_kw
    )
    total_supply = float(np.sum(supply))
    total_demand = float(np.sum(demand_kw))
    latest = df.iloc[-1].to_dict()
    latest['solar_kw'] = round(float(solar_kw.iloc[-1]), 2)
    latest['wind_kw'] = round(float(wind_kw.iloc[-1]), 2)
    return render_template('index.html',
        grid_status=grid_status,
        total_supply=total_supply,
        total_demand=total_demand,
        latest=latest)
