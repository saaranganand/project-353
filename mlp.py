import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load historical data from CSV
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Ensure Close column is numeric
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

# Drop any remaining NaN values
data.dropna(inplace=True)

# Feature Engineering
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=30).std()


# Define target variable (price movement: up/down)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Handle missing values
data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)  # Ensure no remaining NaN values

# Feature selection
features = ['50_MA', '200_MA', 'Daily_Return', 'Volatility']
X = data[features]
y = data['Target']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Train MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot actual vs predicted trends
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Movement', color='blue', alpha=0.6)
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted Movement', color='red', linestyle='dashed', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Stock Movement (Up=1, Down=0)')
plt.title('Actual vs Predicted Stock Movement')
plt.legend()
plt.show()

# Plot stock closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Closing Price', color='black')
plt.plot(data.index, data['50_MA'], label='50-day MA', color='green', linestyle='dashed')
plt.plot(data.index, data['200_MA'], label='200-day MA', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Price (CAD)')
plt.title('VFV.TO Closing Price & Moving Averages')
plt.legend()
plt.show()