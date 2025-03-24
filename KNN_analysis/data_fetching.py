import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt, timedelta

TICKER = "VFV.TO"  # Vanguard S&P 500 Index ETF
START_DATE = "2023-01-01"


def adjust_days(start_date, num_days_before):

    start_date_obj = dt.strptime(start_date, "%Y-%m-%d")

    adjusted_date = start_date_obj - timedelta(days=num_days_before + 30) # Offset by 30 to account for days where no trading was dome

    return adjusted_date.strftime("%Y-%m-%d")


def process_data(data):

    data = data.copy()

    data['50_day_ma'] = data['Close'].rolling(window=50).mean()

    data['daily_return'] = data['Close'].pct_change(1)

    data = data[data.index >= dt.strptime(START_DATE, "%Y-%m-%d")]

    data.dropna(inplace=True)

    return data

def main():

    data = yf.download(TICKER, start=adjust_days(START_DATE, 50), end=dt.now().strftime("%Y-%m-%d"))

    return process_data(data)


if __name__ == "__main__":
    sys.exit(main())