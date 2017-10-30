Playing with some scripts for determining the relative importance of hyperparamters. Ideally done with a tree-based
regressor, so it's human-interpretable. Scikit Learn only works with categorical features as one-hot encoded, so
currently leaning on R. Random Forest (or other ensemble) is preferred, but difficult to to find visualization support;
in which case simple Decision Tree is used.