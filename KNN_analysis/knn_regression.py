import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, \
    ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def KNN_regression(X_train, X_valid, y_train, y_valid):

    params = {
        'n_neighbors': range(3, 20),
        'weights': ['uniform', 'distance'],
        'leaf_size': [20, 25, 30, 35],
        # 'p': [1, 2, 3, 4, 5, 6, 7]
    }
    model = KNeighborsRegressor()

    # KFold = StratifiedKFold(shuffle=True, n_splits=3)
    knn_search = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, n_jobs=-1, refit=True,
                                    n_iter=448)
    knn_search = knn_search.fit(X_train, y_train)
    knn_results = knn_search.cv_results_
    # print(pd.DataFrame(knn_results))
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_valid = ss.transform(X_valid)
    best_params = knn_search.best_params_
    # print(best_params)
    model = KNeighborsRegressor(**best_params).fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    return y_pred, model


