import pandas as pd

START_DATE = "2019-01-01"


def preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess raw stock index data from the given CSV file for MLP regression.

    Steps:
      1. Read CSV, using the first row as column names and drop that row.
      2. Rename 'Price' column to 'Date' if present.
      3. Convert 'Date' column to datetime and set as index.
      4. Convert 'Close' column to numeric.
      5. Calculate 'daily_return' as the percent change of 'Close'.
      6. Calculate 'volatility' as the 30-day rolling std of 'daily_return'.
      7. Create 'target' column as the next day's 'Close'.
      8. Filter rows to dates >= START_DATE.
      9. Drop any rows with missing values.

    Returns:
      A DataFrame indexed by Date with columns ['Close', 'daily_return', 'volatility', 'target'].
    """
    # 1. Read CSV and set first row as header
    raw = pd.read_csv(csv_path, header=None)
    raw.columns = raw.iloc[0]
    raw = raw.iloc[1:].reset_index(drop=True)

    # 2. Rename 'Price' to 'Date' if needed
    if 'Price' in raw.columns:
        raw = raw.rename(columns={'Price': 'Date'})

    # 3. Parse 'Date' and set as index
    raw['Date'] = pd.to_datetime(raw['Date'], format='%Y-%m-%d', errors='coerce')
    raw = raw.dropna(subset=['Date']).set_index('Date')

    # 4. Ensure 'Close' is numeric
    if 'Close' not in raw.columns:
        raise KeyError("'Close' column not found in the dataset.")
    raw['Close'] = pd.to_numeric(raw['Close'], errors='coerce')

    # 5. Calculate daily_return
    raw['daily_return'] = raw['Close'].pct_change()

    # 6. Calculate volatility (30-day rolling std of daily_return)
    raw['volatility'] = raw['daily_return'].rolling(window=30).std()

    # 7. Create target as next day's price
    raw['target'] = raw['Close'].shift(-1)

    # 8. Filter to START_DATE onward
    df = raw[raw.index >= pd.to_datetime(START_DATE)]

    # 9. Drop missing values
    df = df.dropna()

    # Return only the relevant columns
    return df[['Close', 'daily_return', 'volatility', 'target']]