import tensorflow  # this has to be imported _before_ pandas even if not used
import pandas as pd
import os, pdb
import numpy as np
from sqlalchemy.sql import text
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import hypersearch
from data import conn

FOREST = False
PREDICT_PERMUTATIONS = False

# Now, if there are recorded runs in the database, use a random forest to find the best combo
runs = conn.execute('select * from runs').fetchall()
runs = [{**r.hypers, 'target': r.reward} for r in runs]
runs = pd.DataFrame(runs)

# FIXME these shouldn't be in flat, only hydrated
runs.drop(['baseline', 'baseline_optimizer'], axis=1, inplace=True)

# Impute numerical values if possible
runs['dropout'].fillna(0., inplace=True)
runs['gae_lambda'].fillna(1., inplace=True)
for binary in ['keep_last_timestep', 'indicators', 'scale', 'normalize_rewards']:
    runs[binary] = runs[binary].astype(int)

# one-hot encode categorical values
for cat in [
    'step_optimizer.type',
    'net_type',
    'activation',
    'diff',
    'baseline_mode',
]:
    onehot = pd.get_dummies(runs[cat], prefix=cat)
    runs = runs.drop(cat, 1).join(onehot)

X, y = runs.drop('target', 1), runs['target']

# Create model
if FOREST:
    model = RandomForestRegressor()
    hypers = {'n_estimators': [10, 30, 50], 'max_features': [8, 15, 20, 'auto'], 'max_depth': [None, 10, 20],
                 'warm_start': [True, False], 'bootstrap': [True, False]},
else:
    model = DecisionTreeRegressor()
    hypers = {'max_depth': [None, 10, 20], 'max_features': [8, 15, None]}

# Train model w/ cross-val on hypers
model = GridSearchCV(model, hypers, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
model.fit(X.values, y.values)

# Show scores
cvres = model.cv_results_
zipped = sorted(zip(cvres["mean_test_score"], cvres["params"]), key=lambda x: x[0])
for mean_score, params in zipped:
    print(np.sqrt(-mean_score), params)

train_pred = model.predict(X)
print('Train Error', np.sqrt(mean_squared_error(y, train_pred)))


if PREDICT_PERMUTATIONS:
    """
    Predict over cross-joined attr/value combos. Doing so in SQL simpler, see https://stackoverflow.com/a/6683506/362790
    SELECT A AS number, B AS letter FROM
    (VALUES (1), (2), (3)) a(A)
    CROSS JOIN
    (VALUES ('A'), ('B'), ('C')) b(B)
    """
    a, A = 97, 65
    sql_select, sql_body = [], []
    ppo_hypers = hypersearch.hypers['ppo_agent'].copy()
    for k, arr in ppo_hypers.items():
        sql_select.append(f'{chr(A)} as "{k}"')
        vals_ = ', '.join([f"('{v}')" for v in arr])
        sql_body.append(f"(VALUES {vals_}) {chr(a)}({chr(A)})")
        a+=1;A+=1
    sql = "SELECT " + ', '.join(sql_select) + " FROM " + ' CROSS JOIN '.join(sql_body)
    print(sql)

    # Killing system, too many results
    # cartesian = conn.execute(sql).fetchall()


# https://stackoverflow.com/a/40178961/362790
tree = model.best_estimator_.estimators_[1] if FOREST else model.best_estimator_
# export_graphviz(model.best_estimator_.estimators_[1],
export_graphviz(tree,
                feature_names=X.columns,
                filled=True,
                rounded=True)
os.system('dot -Tpng tree.dot -o tree.png')

