from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class CatBoostFeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, num_features_to_select=15, eval_set=None):

        self.num_features_to_select = num_features_to_select
        self.selected_features_indices = []
        self.eval_set = eval_set

    def transform(self, X, y=None):

        _X = X.iloc[:, self.selected_features_indices]

        return _X

    def fit(self, X, y):
        cb = CatBoostClassifier(verbose=False, boosting_type='Ordered', approx_on_full_history=True)

        selected = cb.select_features(X, y, features_for_select= X.columns,
                                      num_features_to_select=self.num_features_to_select,
                                      eval_set=self.eval_set, logging_level='Silent')

        self.selected_features_indices = selected['selected_features']

        return self

