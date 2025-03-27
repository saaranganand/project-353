import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
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
    raise ValueError(f"❌ No data found for {ticker}. Check if the market was open.")

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
    Builds and compiles a simple LSTM model using the last 60 days data to 
    foracst the next days price of VFV.TO
    """
    # The model we are using is sequential Model
    model = Sequential()

    #The input shape is (window_size, 1) where 'window_size' is the number of time steps,
    model.add(Input(shape=(window_size, 1)))

    # This layer processes the sequential data and captures temporal dependencies.
    model.add(LSTM(50))

    # Add a Dense (fully connected) layer with 1 neuron, this layer predicts the final prediction
    model.add(Dense(1))

    # Compile the model using the Adam optimizer and mean squared error (MSE) loss.
    # MSE is commonly used for regression tasks such as predicting a continuous value.
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


# -------------------------------
# DATA SCALING
# -------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(data.values)

# -------------------------------
# WALK-FORWARD SPLIT SETUP
# -------------------------------
# We set an initial training window and a fixed validation window of 6 months
# Then we expand the training set after each fold.
initial_train_end = '2020-12-31'   # initial training ends at 2020-12-31
val_size_days = 180                # ~6 months of validation
max_date = data.index.max()        # last available date

fold_results = []  # Store each fold's results

train_start_date = data.index.min()
train_end_date = pd.to_datetime(initial_train_end)

while True:
    val_start_date = train_end_date + timedelta(days=1)
    val_end_date = val_start_date + timedelta(days=val_size_days - 1)
    
    if val_end_date > max_date:
        # Stop if the validation end date exceeds the available data.
        break
    
    # 1) Training data for this fold (expanding window)
    train_mask = (data.index >= train_start_date) & (data.index <= train_end_date)
    train_data = data.loc[train_mask]
    
    # 2) Validation data for this fold
    val_mask = (data.index >= val_start_date) & (data.index <= val_end_date)
    val_data = data.loc[val_mask]
    
    if len(train_data) < 60 or len(val_data) < 1:
        break  # Not enough data to form sequences
    
    # -------------------------------
    # PREPARE TRAIN/VALIDATION DATA FOR LSTM
    # -------------------------------
    # Use a local scaler to avoid data leakage.
    local_scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = local_scaler.fit_transform(train_data.values)
    
    # Create training sequences
    X_train, y_train = create_sequences(train_scaled, window_size=60)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    # Build and train the LSTM model
    lstm_model = build_lstm(window_size=60)
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Prepare validation sequences.
    # Combine the last 60 days from training with the validation data for continuity.
    combined = pd.concat([train_data.iloc[-60:], val_data])
    combined_scaled = local_scaler.transform(combined.values)
    
    X_val, y_val = create_sequences(combined_scaled, window_size=60)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # Make predictions on the validation sequences.
    val_preds_scaled = lstm_model.predict(X_val)
    val_preds = local_scaler.inverse_transform(val_preds_scaled)
    
    # Map predictions back to their corresponding dates.
    # The sequences start at the 60th entry in the combined dataset.
    val_index = combined.index[60:][:len(val_preds)]
    
    # Create a DataFrame with actual and predicted values.
    # Flatten both the actual values and predictions to ensure they are 1D.
    results_df = pd.DataFrame({
        'Actual': combined.loc[val_index, price_col].values.flatten(),
        'Predicted': val_preds.flatten()
    }, index=val_index)
    fold_results.append(results_df)
    
    # Expand the training window by moving the end date forward.
    train_end_date = val_end_date

# -------------------------------
# EVALUATE WALK-FORWARD RESULTS
# -------------------------------
all_results = pd.concat(fold_results, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data[price_col], label='Historical', color='blue')
plt.plot(all_results.index, all_results['Predicted'], label='Walk-Forward Prediction', color='red')
plt.title(f"{ticker} Walk-Forward Validation (LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (CAD)")
plt.legend()
plt.savefig("WalkForward_LSTM.png")
plt.show()

mse = mean_squared_error(all_results['Actual'], all_results['Predicted'])
rmse = np.sqrt(mse)
print(f"Walk-Forward RMSE: {rmse:.4f}")

# -------------------------------
# FINAL TRAINING ON ALL DATA & FUTURE FORECAST
# -------------------------------
# Retrain on all available data and forecast future prices until the end of 2025.
full_scaler = MinMaxScaler(feature_range=(0, 1))
full_scaled = full_scaler.fit_transform(data.values)

X_full, y_full = create_sequences(full_scaled, window_size=60)
X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))

final_model = build_lstm(window_size=60)
final_model.fit(X_full, y_full, epochs=10, batch_size=32, verbose=1)

# Iteratively forecast future prices.
last_window = full_scaled[-60:].flatten()
n_future = (pd.to_datetime("2025-12-31") - data.index[-1]).days

future_forecast = []
current_window = last_window.copy()
for _ in range(n_future):
    input_seq = current_window.reshape(1, 60, 1)
    pred_scaled = final_model.predict(input_seq)[0, 0]
    future_forecast.append(pred_scaled)
    current_window = np.append(current_window[1:], pred_scaled)

future_forecast = np.array(future_forecast).reshape(-1, 1)
future_forecast = full_scaler.inverse_transform(future_forecast)

future_dates_lstm = pd.date_range(start=data.index[-1] + timedelta(days=1), end="2025-12-31")

plt.figure(figsize=(12, 6))
plt.plot(data.index, data[price_col], label="Historical Price")
plt.plot(future_dates_lstm, future_forecast, label="Final LSTM Forecast (Future)", color='orange')
plt.title(f"{ticker} Future Price Forecast (Trained on All Data)")
plt.xlabel("Date")
plt.ylabel("Price (CAD)")
plt.legend()
plt.savefig("Final_LSTM_Forecast.png")
plt.show()

final_price_2025 = future_forecast[-1][0]
print(f"Forecasted price on 2025-12-31: {final_price_2025:.2f} CAD")
print("✅ Walk-forward validation and final forecast complete.")
