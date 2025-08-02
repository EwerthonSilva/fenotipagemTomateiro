from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.model_selection import cross_val_score


def bestNFeatures(modelo: object, X_train_sc: object, y_train: object) -> tuple[int, Any | None, Any | None]:
    """

    :rtype: tuple[int, Any | None, Any | None]
    """
    best_n_features = 0
    best_score = 0
    best_X_sel = None
    best_sel_features = None
    max_n_features = len(X_train_sc.columns) + 1

    # Check if model has coef_ or feature_importances_ for RFE
    if hasattr(modelo, 'coef_') or hasattr(modelo, 'feature_importances_'):
        for n_features in range(1, max_n_features):
            selector = RFE(modelo, n_features_to_select=n_features, step=1)
            selector = selector.fit(X_train_sc, y_train)

            mask = selector.support_
            features = X_train_sc.columns
            selected_features = features[mask]
            X_sel = X_train_sc[selected_features]

            score = cross_val_score(modelo, X_sel, y_train, cv=10, scoring='r2')

            if np.mean(score) > best_score:
                best_score = np.mean(score)
                best_n_features = n_features
                best_X_sel = X_sel
                best_sel_features = selected_features
    else:
        # If the model does not have coef_ or feature_importances_, use SelectKBest
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X_train_sc, y_train)

        # Try all k values (from 1 to the number of features)
        for n_features in range(1, max_n_features):
            selector.set_params(k=n_features)
            X_sel = selector.transform(X_train_sc)
            selected_features = X_train_sc.columns[selector.get_support()]
            X_sel = pd.DataFrame(X_sel, columns=selected_features)


            score = cross_val_score(modelo, X_sel, y_train, cv=10, scoring='r2')

            if np.mean(score) > best_score:
                best_score = np.mean(score)
                best_n_features = n_features
                best_X_sel = X_sel
                best_sel_features = selected_features

    return best_n_features, best_X_sel, best_sel_features

