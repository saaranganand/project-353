import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load historical data
data = pd.read_csv("../data.csv", index_col=0, parse_dates=True)
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data.dropna(inplace=True)

# Feature Engineering
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=30).std()

# Define target variable
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)

# Feature selection
features = ['50_MA', '200_MA', 'Daily_Return', 'Volatility']
X = data[features]
y = data['Target']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define train-test split
train_data = data[:'2024-12']
test_data = data['2025-01':]
X_train, y_train = train_data[features], train_data['Target']
X_test, y_test = test_data[features], test_data['Target']
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP Classifier with optimized parameters
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.001,
                     learning_rate_init=0.001, max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create directory for plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Visualization
plt.figure(figsize=(12, 7))
plt.plot(test_data.index, y_test, label='Actual', color='blue', alpha=0.6)
plt.plot(test_data.index, y_pred, label='Predicted', color='red', linestyle='dashed', alpha=0.6)
plt.xticks(test_data.index[::len(test_data)//10], rotation=45)  # Show labels at intervals
plt.xlabel('Date')
plt.ylabel('Stock Movement (Up=1, Down=0)')
plt.title('Actual vs Predicted Stock Movement')
plt.legend()
plt.savefig(os.path.join(plot_dir, "actual_vs_predicted.png"))
plt.close()

# Save closing price and moving averages plot
plt.figure(figsize=(12, 7))
plt.plot(data.index, data['Close'], label='Closing Price', color='black')
plt.plot(data.index, data['50_MA'], label='50-day MA', color='green', linestyle='dashed')
plt.plot(data.index, data['200_MA'], label='200-day MA', color='red', linestyle='dashed')
plt.xticks(data.index[::len(data)//10], rotation=45)  # show labels at intervals
plt.xlabel('Date')
plt.ylabel('Price (CAD)')
plt.title('VFV.TO Closing Price & Moving Averages')
plt.legend()
plt.savefig(os.path.join(plot_dir, "closing_price_ma.png"))
plt.close()