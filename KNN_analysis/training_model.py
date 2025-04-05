import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_absolute_error, classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, RocCurveDisplay, auc
import data_fetching


def main():

    data = data_fetching.main()

    # Rebalance data
    no_data = data[data['growth_rate'] == 'No']
    yes_data = data[data['growth_rate'] == 'Yes'].sample(n=no_data.shape[0])

    balanced_Data = pd.concat([yes_data, no_data], ignore_index=True)
    X = balanced_Data.drop(columns=['growth_rate', 'future_close'])

    # Do not count dates for the prediction
    X = X.reset_index(drop=True)

    X = X.dropna()

    X.to_csv('X.csv', index=False)

    y = balanced_Data['growth_rate'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y)

    # best_score = 0
    # best_k = 1
    #
    # model = None

    # Determine the best k value
    # for k in range(1, 20):
    #     model = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights='distance'))
    #     model.fit(X_train, y_train)
    #
    #     score = model.score(X_valid, y_valid)
    #
    #     if score > best_score:
    #         best_score = score
    #         best_k = k

    params = {
        'n_neighbors': range(3, 20),
        'weights': ['uniform', 'distance'],
        'leaf_size': [20, 25, 30, 35],
        # 'p': [1, 2, 3, 4, 5, 6, 7]
    }

    model = KNeighborsClassifier()
    scores = {
        'accuracy': accuracy_score,
        'f1': f1_score
    }

    # KFold = StratifiedKFold(shuffle=True, n_splits=3)
    knn_search = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, n_jobs=-1, refit=True, n_iter=448)
    knn_search = knn_search.fit(X_train, y_train)
    knn_results = knn_search.cv_results_
    print(pd.DataFrame(knn_results))

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_valid = ss.transform(X_valid)
    # print('Best score:', best_score)
    # print('Best K:', best_k)
    best_params = knn_search.best_params_
    print(best_params)
    model = KNeighborsClassifier(**best_params).fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    conf_matrix = confusion_matrix(y_valid, y_pred)

    print(conf_matrix)

    precision = precision_score(y_valid, y_pred, pos_label="Yes")
    recall = recall_score(y_valid, y_pred, pos_label="Yes")
    f1 = f1_score(y_valid, y_pred, pos_label="Yes")

    print(precision, recall, f1)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_valid, y_pred))
    disp.plot(ax=ax)

    title_text = f"F1: {f1*100:.4f}%\n\nClassification Report:\n\n{classification_report(y_valid, y_pred)}"
    ax.set_title(title_text, fontsize=10, loc='left')

    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_cv3.png')

    # y_values = balanced_Data['future_close']
    #
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y_values, test_size=0.2)
    #
    # regression = make_pipeline(StandardScaler(), KNeighborsRegressor(best_k, weights='distance'))
    #
    # regression.fit(X_train, y_train)
    #
    # y_pred = regression.predict(X_valid)
    #
    # print(regression.score(X_valid, y_valid))


if __name__ == '__main__':
    sys.exit(main())