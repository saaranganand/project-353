import yfinance as yf

ticker = "VFV.TO"  # Vanguard S&P 500 Index ETF
data = yf.download(ticker, start="2024-01-01", end="2024-12-31")

print(data.head())

