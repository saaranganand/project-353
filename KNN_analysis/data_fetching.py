import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
from datetime import datetime as dt, timedelta

TICKER = "VFV.TO"  # Vanguard S&P 500 Index ETF
START_DATE = "2017-01-01"

day_adjustment = 50

def adjust_days(start_date, num_days_before):

    start_date_obj = dt.strptime(start_date, "%Y-%m-%d")

    adjusted_date = start_date_obj - timedelta(days=num_days_before + 30) # Offset by 30 to account for days where no trading was dome

    return adjusted_date.strftime("%Y-%m-%d")


def find_growth_rate(x):

    if x['future_close'].values >= x['Close'].values * 1.05:

        return "Yes"

    else:

        return "No"


def process_data(data):

    data = data.copy()

    data['50_day_ma'] = data['Close'].rolling(window=50).mean()

    data['daily_return'] = data['Close'].pct_change(1)

    data['future_close'] = data['Close'].shift(-day_adjustment)

    data = data[data.index >= dt.strptime(START_DATE, "%Y-%m-%d")]

    data.dropna(inplace=True)

    # 'Yes' if there is a 10% increase or more, 'No' if not
    data['growth_rate'] = data.apply(find_growth_rate, axis=1)

    return data

def main():

    data = yf.download(TICKER, start=adjust_days(START_DATE, day_adjustment), end=dt.now().strftime("%Y-%m-%d"))

    return (process_data(data))


if __name__ == "__main__":
    sys.exit(main())