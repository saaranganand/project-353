import pandas as pd
from datetime import datetime as dt, timedelta

START_DATE = "2019-01-01"
date_format = "%m/%d/%Y"

end_date = dt.strptime('12/31/2025', date_format)
current_day = dt.strptime('04/7/2025', date_format)
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

    data = data.iloc[1:].reset_index(drop=True)

    if 'Price' in data.columns:
        data.rename(columns={'Price': 'Date'}, inplace=True)

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='%Y-%m-%d')

    # Set the Date column as the DataFrame index
    data.set_index('Date', inplace=True)

    if 'Close' in data.columns:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    else:
        raise KeyError("'Close' column not found in the dataset.")

    data['daily_return'] = data['Close'].pct_change(1)

    data['future_close'] = data['Close'].shift(-day_adjustment.days)

    data = data[data.index >= pd.to_datetime(START_DATE)]

    data.dropna(inplace=True)

    data['growth_rate'] = data.apply(find_growth_rate, axis=1)

    return data


def main():
    # Adjust header=None, since we have custom logic
    data = pd.read_csv('../data.csv', header=None)
    processed_data = process_data(data)
    return processed_data


if __name__ == '__main__':
    main()
