import os, pdb
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz

from data import conn

# Now, if there are recorded runs in the database, use a random forest to find the best combo
runs = conn.execute('select * from runs').fetchall()

runs = [{**r.hypers, 'target': r.reward} for r in runs]
runs = pd.DataFrame(runs).drop('id', 1)

# one-hot encode categorical values
for cat in ['step_optimizer.type', 'baseline_mode', 'activation', 'diff', 'keep_last_timestep']:
    onehot = pd.get_dummies(runs[cat], prefix=cat)
    runs = runs.drop(cat, axis=1).join(onehot)
# replace None with 0. in some cases
for cat in ['dropout', 'gae_lambda']:
    runs[cat].fillna(0., inplace=True)

# Random Forest
# model = RandomForestRegressor()
# hypers = {'n_estimators': [10, 30, 50], 'max_features': [8, 15, 20, 'auto'], 'max_depth': [None, 10, 20],
#              'warm_start': [True, False], 'bootstrap': [True, False]},
model = DecisionTreeRegressor()
hypers = {'max_depth': [None, 10, 20], 'max_features': [8, 15, None]}

model = GridSearchCV(model, hypers, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
X, y = runs.drop('target', 1), runs['target']
model.fit(X, y)
cvres = model.cv_results_
zipped = sorted(zip(cvres["mean_test_score"], cvres["params"]), key=lambda x: x[0])
for mean_score, params in zipped:
    print(np.sqrt(-mean_score), params)

train_pred = model.predict(X)
print('Train Error', np.sqrt(mean_squared_error(y, train_pred)))

# https://stackoverflow.com/a/40178961/362790
# export_graphviz(model.best_estimator_.estimators_[1],
export_graphviz(model.best_estimator_,
                feature_names=X.columns,
                filled=True,
                rounded=True)
os.system('dot -Tpng tree.dot -o tree.png')

