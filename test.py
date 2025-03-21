import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbol
ticker_symbol = "VFV.TO"

# Download the data
data = yf.download(ticker_symbol, start="2023-01-01", end="2024-12-31")

# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()

# Plot the daily returns over time
plt.figure(figsize=(10, 6))
plt.plot(data['Daily Return'], label='Daily Returns', color='blue')
plt.title(f'Daily Returns for {ticker_symbol} (2024)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
plt.savefig('test.png')

