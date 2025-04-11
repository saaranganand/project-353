import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    classification_report, root_mean_squared_error
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

    # conf_matrix = confusion_matrix(y_valid, y_pred)
    # print(conf_matrix)
    # precision = precision_score(y_valid, y_pred, pos_label="Yes")
    # recall = recall_score(y_valid, y_pred, pos_label="Yes")
    # f1 = f1_score(y_valid, y_pred, pos_label="Yes")
    # print(precision, recall, f1)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_valid, y_pred))
    # disp.plot(ax=ax)
    # title_text = f"F1: {f1 * 100:.4f}%\n\nClassification Report:\n\n{classification_report(y_valid, y_pred)}"
    # ax.set_title(title_text, fontsize=10, loc='left')
    # plt.tight_layout()
    # plt.savefig('plots/confusion_matrix.png')

    # ------------------------------------------------
    # KNN Regressor for Predicting Future Stock Prices
    # ------------------------------------------------

    # # Forming Training and validation data
    # y_values = balanced_Data['future_close'].values
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y_values, test_size=0.2, shuffle=True)
    #
    # # Remove dates from data
    # X_training = X_train.reset_index(drop=True)
    # X_validation = X_valid.reset_index(drop=True)
    #
    # y_pred, knn_regressor = KNN_regression(X_training, X_validation, y_train, y_valid)
    #
    # max = y_valid.max()
    # min = y_valid.min()
    #
    # rmse = root_mean_squared_error(y_valid, y_pred) / (max - min)
    #
    # print(f"The normalized root mean square error is: {rmse}")
    #
    #
    #
    # future_forcast = knn_regressor.predict(forecast_data_no_date)
    #
    # plt.figure(figsize=(10,5))
    #
    # df_plot = pd.DataFrame({
    #     'actual': y_valid,
    #     'predicted': y_pred
    # }, index=X_valid.index)
    #
    # df_plot2 = pd.DataFrame({
    #          'predicted': future_forcast
    #     }, index=forecast_data.index)
    #
    # df_plot = df_plot.sort_index()
    #
    # plt.plot(df_plot.index, df_plot['actual'], 'b.', label='Actual')
    # plt.plot(df_plot.index, df_plot['predicted'], 'r.', label='Predicted')
    # plt.plot(df_plot2.index, df_plot2['predicted'], 'g.', label='Future Forecast')
    #
    # plt.xlabel('Date')
    #
    # plt.ylabel('Future Close')
    # plt.legend(loc='best')
    # plt.savefig("plots/knn_validation_plus_forcast.png")

    # Use regressor to forecast future data from 2024-04-28 - 2025-03-26, the prediction at 2025-03-26 is the predicted
    # closing price by the end of 2025

    # plt.figure(figsize=(10,5))
    #
    # df_plot = pd.DataFrame({
    #     'predicted': future_forcast
    # }, index=forecast_data.index)
    #
    # # Sort by index (e.g., date)
    # df_plot = df_plot.sort_index()
    #
    # plt.plot(df_plot.index, df_plot['predicted'], 'r.', label='Predicted')
    # plt.xlabel('Date')
    #
    # plt.ylabel('Future Close')
    # plt.legend(loc='best')
    # plt.savefig("plots/knn_prediction.png")



if __name__ == '__main__':
    sys.exit(main())