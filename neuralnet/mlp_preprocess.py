#!/usr/bin/env python3
"""
File: mlp_preprocess.py
Description: This script loads data from 'data.csv', preprocesses it for MLP classification,
removing unnecessary features (non-numeric columns), imputing missing values in features,
ensuring there are no missing target values, scaling numerical features,
and then saving the preprocessed dataset to 'data_preprocessed.csv'.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def main():
    # Load the dataset from CSV
    data = pd.read_csv('../data.csv')

    # Optionally drop columns that you do not need (e.g., 'Date' and 'Ticker')
    for col in ['Date', 'Ticker']:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)

    # Explicitly drop rows where the target is missing to avoid issues during training
    data.dropna(subset=['target'], inplace=True)

    # Check if the dataset includes a column for the target
    if 'target' not in data.columns:
        raise ValueError(
            "Target column 'target' is not present in the dataset. Please add a target column for classification.")

    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Select only numeric columns to avoid issues with non-numeric types in features.
    X_numeric = X.select_dtypes(include=[np.number])

    # Impute missing values in features using the mean of each column.
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_numeric)

    # Scale the numeric features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Create a DataFrame from the scaled data with the same feature names.
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)

    # Append the target column back.
    X_scaled_df['target'] = y.values

    # Save the preprocessed data.
    X_scaled_df.to_csv('data_preprocessed.csv', index=False)
    print("Preprocessing complete. Preprocessed data saved to 'data_preprocessed.csv'.")


if __name__ == "__main__":
    main()
