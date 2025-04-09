#!/usr/bin/env python3
"""
File: mlp_regressor_train.py
Description:
    This script uses an MLPRegressor to predict future stock index prices.
    It creates sequences from a 60-day window using two features: "Close" and "volatility".
    The target is the next day's "Close" price.

    Steps:
      1. DATA LOADING & SCALING
      2. TRAIN-VALIDATION SPLIT & SEQUENCE CREATION
      3. MODEL TRAINING
      4. VALIDATION: compute MAE & RMSE, plot Actual vs. Predicted
      5. FUTURE FORECAST: iterative forecast until 2025-12-31
      6. COMBINED PLOT with 4 lines
Plots are saved under the ./plots directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.offsets import BDay

from mlp_regressor_preprocess import preprocess_data

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# ---------------
# PARAMETERS
# ---------------
window_size = 60
feature_cols = ["Close", "volatility"]
target_feature_index = 0  # "Close" is the first feature

# -------------------------------
# DATA LOADING & SCALING
# -------------------------------
data = preprocess_data("../data.csv")
data.sort_index(inplace=True)

# Extract features and scale them
dataset = data[feature_cols].copy()
scaler = MinMaxScaler((0, 1))
scaled_data = scaler.fit_transform(dataset.values)

# Separate scaler for the target ("Close") so we can invert predictions
target_scaler = MinMaxScaler((0, 1))
target_scaler.fit(data[["Close"]].values)

# -------------------------------
# TRAIN-VALIDATION SPLIT & SEQUENCE CREATION
# -------------------------------
training_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_data_len]

x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i - window_size:i, :].flatten())
    y_train.append(train_data[i, target_feature_index])
x_train = np.array(x_train)
y_train = np.array(y_train)

# Validation sequences
test_data = scaled_data[training_data_len - window_size:]
x_valid, y_valid = [], []
for i in range(window_size, len(test_data)):
    x_valid.append(test_data[i - window_size:i, :].flatten())
    y_valid.append(test_data[i, target_feature_index])
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = MLPRegressor(
    hidden_layer_sizes=(35, 25),
    activation='relu',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
print("Training the MLPRegressor model...")
model.fit(x_train, y_train)

# -------------------------------
# VALIDATION / TESTING PREDICTIONS
# -------------------------------
valid_pred = model.predict(x_valid)
# Invert scaling
valid_pred_inv = target_scaler.inverse_transform(valid_pred.reshape(-1, 1))
y_valid_inv = target_scaler.inverse_transform(y_valid.reshape(-1, 1))

mae = mean_absolute_error(y_valid_inv, valid_pred_inv)
rmse = np.sqrt(mean_squared_error(y_valid_inv, valid_pred_inv))
print(f"Validation MAE: {mae:.2f}")
print(f"Validation RMSE: {rmse:.2f}")

# -------------------------------
# PLOT 1: Validation Actual vs. Predicted
# -------------------------------
validation_dates = data.index[training_data_len: training_data_len + len(y_valid)]

plt.figure(figsize=(10, 6))
plt.plot(validation_dates, y_valid_inv, label="Actual Price", marker="o")
plt.plot(validation_dates, valid_pred_inv, label="Predicted Price", marker="x")
plt.xlabel("Date")
plt.ylabel("Close Price (CAD)")
plt.title("Actual vs. Predicted Price (Validation)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/mlp_validation_plot.png")
plt.show()

# -------------------------------
# FUTURE FORECAST UNTIL DECEMBER 31, 2025
# -------------------------------
last_date = data.index[-1]
forecast_start = last_date + BDay(1)
forecast_end = pd.to_datetime("2025-12-31")
future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='B')
n_future = len(future_dates)

# Prepare iterative forecasting
last_window = scaled_data[-window_size:, :]
current_window = last_window.flatten()

future_preds = []
for _ in range(n_future):
    inp = current_window.reshape(1, window_size * len(feature_cols))
    pred_scaled = model.predict(inp)[0]
    future_preds.append(pred_scaled)
    # update window: drop oldest day and append [predicted_close, last_volatility]
    last_vol = current_window[-1]  # volatility of last day
    new_feats = np.array([pred_scaled, last_vol])
    current_window = np.append(current_window[2:], new_feats)

future_preds = np.array(future_preds).reshape(-1, 1)
future_preds_inv = target_scaler.inverse_transform(future_preds)

# -------------------------------
# PLOT 2: Combined Forecast Plot
# -------------------------------
plt.figure(figsize=(12, 6))

# a) Training Actual
train_dates = data.index[:training_data_len]
plt.plot(train_dates, data["Close"].iloc[:training_data_len], label="Training Actual", color="blue")

# b) Validation Actual
plt.plot(validation_dates, data["Close"].iloc[training_data_len:training_data_len + len(y_valid)],
         label="Validation Actual", color="green")

# c) Validation Predictions
plt.plot(validation_dates, valid_pred_inv, label="Validation Predictions", color="orange", linestyle="--")

# d) Future Forecast
plt.plot(future_dates, future_preds_inv, label="Future Forecast", color="purple", linestyle="-.")

plt.xlabel("Date")
plt.ylabel("Close Price (CAD)")
plt.title("MLP Stock Price Prediction & Future Forecast")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/price_forecast_full.png")
plt.show()

# -------------------------------
# PRINT FINAL FUTURE PRICE
# -------------------------------
print(f"Forecasted price on 2025-12-31: {future_preds_inv[-1, 0]:.2f} CAD")