"""
Flask dashboard exposing routes for visualization and simple controls.
Uses Plotly for interactive charts embedded in HTML.
"""
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import plotly.express as px
from api.data_collection import load_sample
from forecasting.forecasting import solar_power_prediction, wind_power_prediction

app = Flask(__name__)

TEMPLATE = """
<html>
  <head>
    <title>AI Weather Energy Dashboard</title>
  </head>
  <body>
    <h1>AI Weather & Energy Dashboard</h1>
    <div>
      <form method="get">
        <button name="live" value="0">Load Sample Data</button>
      </form>
    </div>
    <div>
      {{ solar_div | safe }}
    </div>
    <div>
      {{ wind_div | safe }}
    </div>
  </body>
</html>
"""

@app.route('/')
def index():
    # Load sample data for demo
    df = load_sample('data/sample_data.csv')
    # Prepare simple plots
    solar_kw = solar_power_prediction(df, model_path=None)
    wind_kw = wind_power_prediction(df, model_path=None)

    df_plot = df.copy()
    df_plot['solar_kw'] = solar_kw
    df_plot['wind_kw'] = wind_kw

    fig_solar = px.line(df_plot, x='datetime', y='solar_kw', title='Estimated Solar Generation (kW)')
    fig_wind = px.line(df_plot, x='datetime', y='wind_kw', title='Estimated Wind Generation (kW)')

    solar_div = fig_solar.to_html(full_html=False)
    wind_div = fig_wind.to_html(full_html=False)
    return render_template_string(TEMPLATE, solar_div=solar_div, wind_div=wind_div)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to accept a JSON payload of features and return predictions."""
    payload = request.json
    df = pd.DataFrame(payload)
    solar = solar_power_prediction(df, model_path=None)
    wind = wind_power_prediction(df, model_path=None)
    return jsonify({'solar_kw': solar.tolist() if hasattr(solar, 'tolist') else None,
                    'wind_kw': wind.tolist() if hasattr(wind, 'tolist') else None})
