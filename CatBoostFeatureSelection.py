import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class CatBoostFeatureSelection(BaseEstimator, TransformerMixin):
    """
       A scikit-learn compatible feature selection transformer using CatBoostClassifier.

       This transformer selects a subset of features based on their importance using the CatBoostClassifier algorithm.

       Parameters:
       ----------
       num_features_to_select : int, optional (default=15)
           The number of features to select.

       eval_set : tuple or None, optional (default=None)
           The evaluation set as a tuple of (X_eval, y_eval) to be used during feature selection.
           If None, the training set will be used.

       Attributes:
       ----------
       num_features_to_select : int
           The number of features to select.

       selected_features_indices : list
           The indices of the selected features.

       eval_set : tuple or None
           The evaluation set as a tuple of (X_eval, y_eval) used during feature selection.

       Methods:
       ----------
       transform(X, y=None):
           Transform the input DataFrame X to include only the selected features.

       fit(X, y):
           Fit the transformer by selecting the most important features based on the CatBoostClassifier.

       """

    def __init__(self, num_features_to_select=15, eval_set=None):
        """
        Initialize the CatBoostFeatureSelection transformer.

        Parameters:
        ----------
        num_features_to_select : int, optional (default=15)
            The number of features to select.

        eval_set : tuple or None, optional (default=None)
            The evaluation set as a tuple of (X_eval, y_eval) to be used during feature selection.
            If None, the training set will be used.
        """
        self.num_features_to_select = num_features_to_select
        self.selected_features_indices = []
        self.eval_set = eval_set

    def transform(self, X, y=None):
        """
        Transform the input DataFrame X to include only the selected features.

        Parameters:
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            The input DataFrame to transform.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values. This parameter is ignored and only included for compatibility.

        Returns:
        ----------
        X_transform : pandas DataFrame, shape (n_samples, num_features_to_select)
            The transformed DataFrame with only the selected features.

        Raises:
        ----------
        ValueError:
            If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        x_transformed = X.iloc[:, self.selected_features_indices]

        return x_transformed

    def fit(self, X, y):
        """
        Fit the transformer by selecting the most important features based on the CatBoostClassifier.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            The input array-like object (pandas DataFrame, NumPy array) to fit.

        y : array-like, shape (n_samples,)
            The target values.

        Returns:
        ----------
        self : CatBoostFeatureSelection
            Returns the instance itself.

        Raises:
        ----------
        ValueError:
            If X is not a pandas DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        cb = CatBoostClassifier(verbose=False, boosting_type='Ordered', approx_on_full_history=True)

        selected = cb.select_features(X, y, features_for_select=X.columns,
                                      num_features_to_select=self.num_features_to_select,
                                      eval_set=self.eval_set, logging_level='Silent')

        self.selected_features_indices = selected['selected_features']

        return self

