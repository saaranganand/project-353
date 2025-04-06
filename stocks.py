import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plot styles
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Define the stock ticker
ticker = "VFV.TO"

# Set up start and end dates
start_date = "2019-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date, progress=False, ignore_tz=True)

# Check if data is available
if data.empty:
    print(f"❌ No data found for {ticker}. Check if the market was open.")
else:
    # Print available columns to check what data is available
    print("Available columns:", data.columns)

    # Check if 'Adj Close' exists; if not, use 'Close' instead
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    # Calculate key metrics
    data["Daily Return"] = data[price_col].pct_change() * 100  # Daily return in %
    data["50-day MA"] = data[price_col].rolling(window=50).mean()  # 50-day moving average
    data["200-day MA"] = data[price_col].rolling(window=200).mean()  # 200-day moving average
    data["Volatility"] = data["Daily Return"].rolling(window=30).std()  # 30-day rolling volatility

    # Show first few rows
    print(data.head())

    # Save to CSV (optional)
    data.to_csv(f"{ticker}_2024_data.csv")
    print(f"✅ Data saved as '{ticker}_2024_data.csv'.")

    # Plot the stock price trend
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[price_col], label="Stock Price", linewidth=2)
    plt.plot(data.index, data["50-day MA"], label="50-day Moving Average", linestyle="--")
    plt.plot(data.index, data["200-day MA"], label="200-day Moving Average", linestyle="--")
    plt.title(f"{ticker} Price Trend (2019-2025)")
    plt.xlabel("Date")
    plt.ylabel("Price (CAD)")
    plt.legend()
    plt.savefig("Stocks.png")

    # Plot daily return
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Daily Return"], label="Daily Return", color="purple")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(f"{ticker} Daily Returns (2019-2025)")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.legend()
    plt.savefig("Daily_Return.png")
    plt.show()

    # Plot volatility (rolling standard deviation of daily returns)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Volatility"], label="30-day Rolling Volatility", color="red")
    plt.title(f"{ticker} Volatility (2019-2025)")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Std Dev of Daily Return)")
    plt.legend()
    plt.savefig("Volatility.png")
    plt.show()
