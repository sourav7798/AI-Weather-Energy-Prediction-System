"""
Simple LSTM implementation using TensorFlow Keras for time-series prediction.
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm(input_shape, units=50):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train, y_train, X_val, y_val, save_path, seq_len=24, epochs=50, batch_size=32):
    Xs_train, ys_train = create_sequences(X_train, y_train, seq_len)
    Xs_val, ys_val = create_sequences(X_val, y_val, seq_len)
    model = build_lstm((Xs_train.shape[1], Xs_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(Xs_train, ys_train, validation_data=(Xs_val, ys_val), epochs=epochs, batch_size=batch_size, callbacks=[es])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model
