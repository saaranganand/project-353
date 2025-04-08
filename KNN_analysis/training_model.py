import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    classification_report, root_mean_squared_error
from sklearn.model_selection import train_test_split
import data_fetching
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

    # Do not count dates for the prediction
    # X = X.reset_index(drop=True)
    #
    # X = X.dropna()
    #
    # y = balanced_Data['growth_rate'].values
    #
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)
    #
    # y_pred = KNN_classifier(X_train, X_valid, y_train, y_valid)
    #
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

    y_values = balanced_Data['future_close'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y_values, test_size=0.2)

    X_training = X_train.reset_index(drop=True)
    X_validation = X_valid.reset_index(drop=True)

    y_pred = KNN_regression(X_training, X_validation, y_train, y_valid)

    max = y_valid.max()
    min = y_valid.min()

    rmse = root_mean_squared_error(y_valid, y_pred) / (max - min)

    print(f"The normalized root mean square error is: {rmse}")
    plt.figure(figsize=(10,5))

    plt.plot(X_valid.index, y_valid, 'b.', label='Actual Future Close')
    plt.plot(X_valid.index, y_pred, 'r.', label='Predicted Future Close')
    plt.xlabel('Date')
    plt.ylabel('Future Close')
    plt.legend(loc='best')
    plt.savefig("plots/knn_validation.png")

if __name__ == '__main__':
    sys.exit(main())