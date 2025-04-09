import pandas as pd
from datetime import datetime as dt, timedelta


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

    data.dropna(inplace=True)

    return data


def main():
    # Adjust header=None, since we have custom logic
    data = pd.read_csv('future_forecast.csv', header=None)
    processed_data = process_data(data)
    return processed_data


if __name__ == '__main__':
    main()