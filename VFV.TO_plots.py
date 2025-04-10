import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styles
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Ensure the "Graphs" directory exists
os.makedirs("Graphs", exist_ok=True)

# Path to the local CSV file
csv_path = "data.csv"

# Read the CSV file:
data = pd.read_csv(csv_path, skiprows=1, header=0, parse_dates=False)
print("Raw available columns:", data.columns.tolist())

new_names = {
    data.columns[0]: "Date",
    data.columns[1]: "Open",
    data.columns[2]: "High",
    data.columns[3]: "Low",
    data.columns[4]: "Close"
}
data.rename(columns=new_names, inplace=True)
print("Renamed columns:", data.columns.tolist())

# Convert the Date column to datetime.
data["Date"] = pd.to_datetime(data["Date"], errors="coerce", format="%Y-%m-%d")
if data["Date"].isnull().all():
    raise ValueError("Date conversion failed for all rows. Check the CSV date format.")

# Set Date as the index.
data.set_index("Date", inplace=True)

# Optionally, filter data from 2019 onward.
data = data[data.index >= pd.to_datetime("2019-01-01")]

# Use the 'Close' column for price calculations.
price_col = "Close"

# Calculate Daily Return (in %)
data["Daily Return"] = data[price_col].pct_change() * 100

# Calculate 50-day Moving Average
data["50-day MA"] = data[price_col].rolling(window=50).mean()

# Calculate 200-day Moving Average
data["200-day MA"] = data[price_col].rolling(window=200).mean()

# Calculate 30-day Rolling Volatility (std. dev. of Daily Return)
data["Volatility"] = data["Daily Return"].rolling(window=30).std()

# Print first few rows for verification.
print(data.head())

# -------------------------------
# Plot 1: Price Trend
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[price_col], label="Stock Price", linewidth=2)
plt.plot(data.index, data["50-day MA"], label="50-day MA", linestyle="--")
plt.plot(data.index, data["200-day MA"], label="200-day MA", linestyle="--")
plt.title("Stock Price Trend (2019-2025)")
plt.xlabel("Date")
plt.ylabel("Price (CAD)")
plt.legend()
plt.tight_layout()
plt.savefig("Graphs/Stocks.png")
plt.close()

# -------------------------------
# Plot 2: Daily Return
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Daily Return"], label="Daily Return", color="purple")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Daily Return (2019-2025)")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.legend()
plt.tight_layout()
plt.savefig("Graphs/Daily_Return.png")
plt.close()

# -------------------------------
# Plot 3: Volatility
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Volatility"], label="30-day Rolling Volatility", color="red")
plt.title("Volatility (2019-2025)")
plt.xlabel("Date")
plt.ylabel("Volatility (Std Dev of Daily Return)")
plt.legend()
plt.tight_layout()
plt.savefig("Graphs/Volatility.png")
plt.close()

plt.show()