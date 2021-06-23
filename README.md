# CatBoostFeatureSelection
Feature Selection using CatBoost select_features funcrtion. Built as Sklearn Transfomer, with option to use evaluation set.



Usage example:

feature_selection = CatBoostFeatureSelection(eval_set=(x_validation, y_validation))
feature_selection = feature_selection.fit(x_train, y_train)
x_train = feature_selection.transform(x_train)
x_test = feature_selection.transform(x_test)
