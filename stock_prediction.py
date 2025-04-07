import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# -------------------------------
# SETTINGS & DATA DOWNLOAD
# -------------------------------
ticker = "VFV.TO"
start_date = "2019-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

data = yf.download(ticker, start=start_date, end=end_date, progress=False, ignore_tz=True)
if data.empty:
    raise ValueError(f"No data found for {ticker}. Check if the market was open.")

# Use 'Adj Close' if available; otherwise, use 'Close'
price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
data = data[[price_col]].copy()
data.index = pd.to_datetime(data.index)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def create_sequences(dataset, window_size=60):
    """
    Creates sequences of length 'window_size' for LSTM training.
    Returns (X, y) as NumPy arrays.
    """
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i - window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def build_lstm(window_size=60):
    """
    Builds and compiles a simple LSTM model using the last 'window_size' days 
    to forecast the next day's price.
    """
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# -------------------------------
# DATA SCALING & PREPARATION
# -------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(data.values)

# Create sequences using the entire available dataset
X, y = create_sequences(scaled_values, window_size=60)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------------
# TRAINING ON ALL DATA UP TO CURRENT DATE
# -------------------------------
model = build_lstm(window_size=60)
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# -------------------------------
# FUTURE FORECAST UNTIL THE END OF 2025
# -------------------------------
# Get the last 60 days from the scaled data to start the iterative forecast
last_window = scaled_values[-60:].flatten()
n_future = (pd.to_datetime("2025-12-31") - data.index[-1]).days

future_forecast = []
current_window = last_window.copy()

for _ in range(n_future):
    input_seq = current_window.reshape(1, 60, 1)
    pred_scaled = model.predict(input_seq)[0, 0]
    future_forecast.append(pred_scaled)
    current_window = np.append(current_window[1:], pred_scaled)

future_forecast = np.array(future_forecast).reshape(-1, 1)
future_forecast = scaler.inverse_transform(future_forecast)

future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), end="2025-12-31")

plt.figure(figsize=(12, 6))
plt.plot(data.index, data[price_col], label="Historical Price")
plt.plot(future_dates, future_forecast, label="LSTM Forecast (Future)", color='orange')
plt.title(f"{ticker} Future Price Forecast (Trained on All Data)")
plt.xlabel("Date")
plt.ylabel("Price (CAD)")
plt.legend()
plt.show()

final_price_2025 = future_forecast[-1][0]
print(f"Forecasted price on 2025-12-31: {final_price_2025:.2f} CAD")
