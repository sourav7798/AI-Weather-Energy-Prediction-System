"""
Data collection module: fetches live weather data from OpenWeatherMap
or reads sample CSV. Saves data to CSV and handles missing values.
"""
import os
import requests
import pandas as pd
from datetime import datetime

API_URL = 'https://api.openweathermap.org/data/2.5/weather'

def fetch_current_weather(lat, lon, api_key=None):
    """Fetch current weather for a lat/lon from OpenWeatherMap.

    Returns a dict with selected fields or raises an exception.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        raise RuntimeError('OPENWEATHER_API_KEY not set')

    params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}
    resp = requests.get(API_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Map fields
    out = {
        'datetime': datetime.utcfromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': data['main'].get('temp'),
        'humidity': data['main'].get('humidity'),
        'pressure': data['main'].get('pressure'),
        'wind_speed': data.get('wind', {}).get('speed'),
        'clouds': data.get('clouds', {}).get('all'),
        'solar_irradiance': None  # placeholder: OpenWeather does not provide irradiance
    }
    return out

def append_to_csv(record, csv_path):
    """Append a single record (dict) to CSV; create if missing."""
    df = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def load_sample(path):
    """Load the provided sample CSV dataset."""
    return pd.read_csv(path, parse_dates=['datetime'])
