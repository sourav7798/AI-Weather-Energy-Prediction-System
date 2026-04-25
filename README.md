# AI-Based Weather Prediction System for Renewable Energy Generation Forecasting and Grid Optimization

## Abstract
This project builds an end-to-end system that ingests real-time weather data, predicts weather variables, forecasts solar and wind generation, and performs a basic grid optimization to balance supply and demand. It includes a Flask dashboard for visualization and controls.

## Structure
See the repository folders for modules: `api`, `preprocessing`, `models`, `forecasting`, `optimization`, `dashboard`, `data`.

## Setup
1. Create a Python 3.8+ venv and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
set OPENWEATHER_API_KEY=your_api_key_here
```

## Run
- To run the dashboard locally:

```bash
python main.py
```

## Files of interest
- `api/data_collection.py` — fetches live data and saves CSV
- `preprocessing/preprocess.py` — cleaning and feature prep
- `models/train_models.py` — trains LR and RF
- `models/lstm_model.py` — trains LSTM
- `forecasting/forecasting.py` — energy forecasting
- `optimization/grid_optimization.py` — simple grid optimizer
- `dashboard/app.py` — Flask dashboard

## Notes
- This is a final-year project scaffold. Replace API keys and tune model hyperparameters for production.
