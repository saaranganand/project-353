import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    classification_report, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import data_fetching
from KNN_analysis import process_forecast_data
from KNN_analysis.knn_classifier import KNN_classifier
from KNN_analysis.knn_regression import KNN_regression


def main():

    data = data_fetching.main()

    # Rebalance data
    no_data = data[data['growth_rate'] == 'No']
    yes_data = data[data['growth_rate'] == 'Yes'].sample(n=no_data.shape[0])

    balanced_Data = pd.concat([yes_data, no_data], ignore_index=False)

    balanced_data = balanced_Data.sort_index()

    # print(balanced_data.head())

    X = balanced_data.drop(columns=['growth_rate', 'future_close'])

    forecast_data = process_forecast_data.main()

    forecast_data_no_date = forecast_data.reset_index(drop=True)
    # print(forecast_data)
    ss = StandardScaler()

    forecast_data_no_date = ss.fit_transform(forecast_data_no_date)

    # ------------------------------------------
    # KNN Classifier for Predicting Stock Growth
    # ------------------------------------------

    # Do not count dates for the prediction
    X = X.reset_index(drop=True)

    X = X.dropna()

    y = balanced_Data['growth_rate'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)

    y_pred, knn_classifier = KNN_classifier(X_train, X_valid, y_train, y_valid)

    future_forecast_data = knn_classifier.predict(forecast_data_no_date)

    print(future_forecast_data)

    conf_matrix = confusion_matrix(y_valid, y_pred)
    print(conf_matrix)
    precision = precision_score(y_valid, y_pred, pos_label="Yes")
    recall = recall_score(y_valid, y_pred, pos_label="Yes")
    f1 = f1_score(y_valid, y_pred, pos_label="Yes")
    print(precision, recall, f1)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_valid, y_pred))
    disp.plot(ax=ax)
    title_text = f"F1: {f1 * 100:.4f}%\n\nClassification Report:\n\n{classification_report(y_valid, y_pred)}"
    ax.set_title(title_text, fontsize=10, loc='left')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')

    # ------------------------------------------------
    # KNN Regressor for Predicting Future Stock Prices
    # ------------------------------------------------

    # Forming Training and validation data
    y_values = balanced_Data['future_close'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y_values, test_size=0.2, shuffle=True)

    # Remove dates from data
    X_training = X_train.reset_index(drop=True)
    X_validation = X_valid.reset_index(drop=True)

    y_pred, knn_regressor = KNN_regression(X_training, X_validation, y_train, y_valid)

    rmse = root_mean_squared_error(y_valid, y_pred)

    print(f"The root mean square error is: {rmse}")

    mae = mean_absolute_error(y_valid, y_pred)

    print(f"The mean absolute error is: {mae}")

    # Use regressor to forecast future data from 2024-04-27 - 2025-03-26, the prediction at 2025-03-26 is the predicted
    # closing price by the end of 2025

    future_forcast = knn_regressor.predict(forecast_data_no_date)

    print(future_forcast[-1])
    plot_validation_data(X_valid, y_valid, y_pred)

    plot_future_forecast(X_valid, forecast_data, future_forcast, y_pred, y_valid)


def plot_future_forecast(X_valid, forecast_data, future_forcast, y_pred, y_valid):

    plt.figure(figsize=(10, 5))
    df_plot = pd.DataFrame({
        'actual': y_valid,
        'predicted': y_pred
    }, index=X_valid.index)

    df_plot2 = pd.DataFrame({
        'predicted': future_forcast
    }, index=forecast_data.index)

    df_plot = df_plot.sort_index()
    df_plot2 = df_plot2.sort_index()

    plt.title("KNN Regressor Future Closing Price Prediction & Future Forecast")

    plt.plot(df_plot.index, df_plot['actual'], 'b.', label='Actual')
    plt.plot(df_plot.index, df_plot['predicted'], 'r.', label='Predicted')
    plt.plot(df_plot2.index, df_plot2['predicted'], 'g.', label='Future Forecast')

    plt.xlabel('Date')
    plt.ylabel('Future Close')
    plt.legend(loc='best')

    plt.savefig("plots/knn_validation_plus_forcast.png")


def plot_validation_data(X_valid, y_valid, y_pred):

    plt.figure(figsize=(10, 5))
    df_plot = pd.DataFrame({
        'actual': y_valid,
        'predicted': y_pred
    }, index=X_valid.index)

    df_plot = df_plot.sort_index()

    plt.title('Validation Data vs Predicted Data')
    plt.plot(df_plot.index, df_plot['actual'], 'b.', label='Actual')
    plt.plot(df_plot.index, df_plot['predicted'], 'r.', label='Predicted'
                                                              '')
    plt.xlabel('Date')
    plt.ylabel('Future Close')
    plt.legend(loc='best')

    plt.savefig("plots/knn_validation.png")


if __name__ == '__main__':
    sys.exit(main())