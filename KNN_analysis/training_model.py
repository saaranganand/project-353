import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import data_fetching


def main():

    data = data_fetching.main()

    yes_data = data[data['growth_rate'] == 'Yes']

    no_data = data[data['growth_rate'] == 'No']

    print(len(yes_data), len(no_data))

    balanced_Data = pd.concat([yes_data, no_data], ignore_index=True)

    X = balanced_Data.drop(columns=['growth_rate', 'future_close', '50_day_ma', 'daily_return'])

    X = X.reset_index(drop=True)

    y = balanced_Data['growth_rate'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    best_score = 0
    best_k = 1

    model = None

    for k in range(1, 20):
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(k))
        model.fit(X_train, y_train)

        score = model.score(X_valid, y_valid)

        if score > best_score:
            best_score = score
            best_k = k

    print('Best score:', best_score)
    print('Best K:', best_k)

    y_pred = model.predict(X_valid)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    print(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel()

    false_positive_rate = fp / (fp + fn)
    false_negative_rate = fn / (fp + fn)
    precision = precision_score(y_valid, y_pred, pos_label="Yes")
    recall = recall_score(y_valid, y_pred, pos_label="Yes")
    true_negative_rate = tn / (tn + fp)
    f1 = f1_score(y_valid, y_pred, pos_label="Yes")
    print(precision, recall, f1)
    # model = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
    #
    # model.fit(X_train, y_train)
    #
    # y_pred = model.predict(X_valid)
    #
    # conf_matrix = confusion_matrix(y_valid, y_pred)
    # print(conf_matrix)
    #
    # print(model.score(X_train, y_train))
    # print(model.score(X_valid, y_valid))


if __name__ == '__main__':
    sys.exit(main())