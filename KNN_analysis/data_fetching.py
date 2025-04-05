import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta

START_DATE = "2019-01-01"
date_format = "%m/%d/%Y"

end_date = dt.strptime('12/31/2025', date_format)
current_day = dt.now()
day_adjustment = end_date - current_day


def find_growth_rate(x):
    if x['future_close'] >= x['Close'] * 1.10:
        return "Yes"
    else:
        return "No"


def process_data(data):
    data = data.copy()

    # Use the first row as column names
    data.columns = data.iloc[0]
    # Drop that row
    data = data.iloc[1:].reset_index(drop=True)

    # Rename the 'Price' column to 'Date'
    if 'Price' in data.columns:
        data.rename(columns={'Price': 'Date'}, inplace=True)

    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='%Y-%m-%d')

    # Set the Date column as the DataFrame index
    data.set_index('Date', inplace=True)

    # Convert Close to numeric
    if 'Close' in data.columns:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    else:
        raise KeyError("'Close' column not found in the dataset.")

    # Calculate daily return
    data['daily_return'] = data['Close'].pct_change(1)

    # Shift close price for future close
    data['future_close'] = data['Close'].shift(-day_adjustment.days)

    # Filter rows starting from START_DATE
    data = data[data.index >= pd.to_datetime(START_DATE)]

    # Drop rows where 'Close' or 'future_close' is missing
    data.dropna(inplace=True)

    # Check for 10% growth
    data['growth_rate'] = data.apply(find_growth_rate, axis=1)

    return data


def main():
    # Adjust header=None, since we have custom logic
    data = pd.read_csv('../data.csv', header=None)
    processed_data = process_data(data)
    return processed_data


if __name__ == '__main__':
    main()
