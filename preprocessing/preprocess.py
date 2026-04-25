"""
Preprocessing utilities: cleaning, imputation, scaling, feature selection,
and time-series train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_missing(df):
    """Handle missing values: simple forward-fill then median impute."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df = df.fillna(method='ffill')
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def feature_engineering(df):
    """Add time features and simple interactions useful for models."""
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    # Simple solar angle proxy: higher irradiance during 6-18
    df['is_day'] = df['hour'].between(6,18).astype(int)
    # Example interaction
    df['temp_times_irradiance'] = df['temperature'] * df.get('solar_irradiance', 0)
    return df

def scale_features(X_train, X_test):
    """Standard-scale features and return scaler for inverse transform."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def time_series_train_test(df, target_col, test_size=0.2):
    """Split time-series data into train/test without shuffling.

    Returns X_train, X_test, y_train, y_test and the full feature names.
    """
    df = df.sort_values('datetime')
    features = [c for c in df.columns if c not in ['datetime', target_col]]
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    X_train = train[features]
    X_test = test[features]
    y_train = train[target_col]
    y_test = test[target_col]
    return X_train, X_test, y_train, y_test, features
