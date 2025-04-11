import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set up plot style
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# -------------------------------
# SETTINGS & DATA DOWNLOAD
# -------------------------------
ticker = "VFV.TO"
start_date = "2019-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

print("Downloading data for", ticker)
data = pd.read_csv("../data.csv", skiprows=[1, 2], parse_dates=[0], index_col=0)
if data.empty:
    raise ValueError(f"No data found for {ticker}. Check the ticker symbol and date range.")

# Use 'Adj Close' if available; otherwise, use 'Close'
price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
data = data[[price_col]].copy()

# -------------------------------
# DATA PREPROCESSING & SCALING
# -------------------------------
dataset = data.copy()  # keep copy for inverse scaling later
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset.values)

# -------------------------------
# TRAIN-VALIDATION SPLIT & SEQUENCE CREATION
# -------------------------------
window_size = 60
training_data_len = int(len(scaled_data) * 0.8)

# Create training sequences
train_data = scaled_data[:training_data_len]
x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -------------------------------
# BUILD & TRAIN THE IMPROVED LSTM MODEL
# -------------------------------
model = Sequential()

# First LSTM layer with dropout
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer with dropout
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# Dense layers
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
print("Training the model...")

# Use EarlyStopping and ReduceLROnPlateau to control overfitting and adjust learning rate
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Prepare validation data from training_data_len - window_size for proper sequence
test_data = scaled_data[training_data_len - window_size:]
x_test, y_test = [], dataset.values[training_data_len:]
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i - window_size:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, scaler.transform(y_test)),  # Compare in scaled units
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -------------------------------
# VALIDATION / TESTING PREDICTIONS
# -------------------------------
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print("Root Mean Squared Error (RMSE) on validation set:", rmse)

# -------------------------------
# FUTURE FORECAST UNTIL DECEMBER 31, 2025
# -------------------------------
last_window = scaled_data[-window_size:]
current_window = last_window.flatten()

n_future_days = (pd.to_datetime("2025-12-31") - data.index[-1]).days
future_predictions = []

for _ in range(n_future_days):
    current_input = current_window.reshape(1, window_size, 1)
    pred = model.predict(current_input)[0, 0]
    future_predictions.append(pred)
    current_window = np.append(current_window[1:], pred)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), end="2025-12-31")

# -------------------------------
# PLOTTING
# -------------------------------
# Split original data for plotting
train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16, 6))
plt.title('Improved LSTM Stock Price Prediction & Future Forecast')
plt.xlabel('Date', fontsize=18)
plt.ylabel(f'{price_col} Price (CAD)', fontsize=18)

# Plot the training data
plt.plot(train.index, train[price_col], label='Train', color='blue')

# Plot the actual validation data
plt.plot(valid.index, valid[price_col], label='Validation Actual', color='green')

# Plot the validation predictions
plt.plot(valid.index, valid['Predictions'], label='Validation Predictions', color='orange', alpha=0.7)

# Plot the future forecast
plt.plot(future_dates, future_predictions, label='Future Forecast', color='purple', linewidth=2)

plt.legend(loc='lower right')
#plt.show()
plt.savefig("LSTM_Future_Forcast.png")

# -------------------------------
# PRINT FINAL FUTURE PRICE
# -------------------------------
final_price = future_predictions[-1][0]
print(f"Forecasted price on 2025-12-31: {final_price:.2f} CAD")
