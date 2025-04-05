import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt, timedelta

# TICKER = "VFV.TO"  # Vanguard S&P 500 Index ETF

START_DATE = "2019-01-01"
date_format = "%m/%d/%Y"


end_date = dt.strptime('12/31/2025', date_format)
current_day = dt.now()
day_adjustment = end_date - current_day


def adjust_days(start_date, num_days_before):
    start_date_obj = dt.strptime(start_date, "%Y-%m-%d")

    adjusted_date = start_date_obj - timedelta(days=num_days_before)
    return adjusted_date.strftime("%Y-%m-%d")


def find_growth_rate(x):
    if x['future_close'].values >= x['Close'].values * 1.10:

        return "Yes"

    else:

        return "No"


def process_data(data):
    data = data.copy()

    # Rename and format columns
    # data.columns = data.iloc[1]
    # data = data.iloc[2:].reset_index(drop=True)

    print(data)
    # Convert date and close price columns to correct formats
    # data['Date'] = pd.to_datetime(data['Date'])
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # data.set_index('Date', inplace=True)

    # Calculate daily return
    data['daily_return'] = data['Close'].pct_change(1)

    # Shift close price for future close
    data['future_close'] = data['Close'].shift(-day_adjustment.days)

    data = data[data.index >= pd.to_datetime(START_DATE)]

    data.dropna(subset=['Close', 'future_close'], inplace=True)

    # Check for 10% growth
    data['growth_rate'] = data.apply(find_growth_rate, axis=1)

    return data


def main():
    # data = yf.download(TICKER, start=START_DATE, end=current_day.strftime('%Y-%m-%d'))
    data = pd.read_csv('../data.csv', index_col=0, parse_dates=True)
    return (process_data(data))


if __name__ == "__main__":
    sys.exit(main())
