# import tensorflow  # this has to be imported _before_ pandas even if not used
from hypersearch import hypers as hypers_dict # actually, handle the ordering in hypersearch.py & import that here
import pandas as pd
import os, pdb, json, argparse
import numpy as np
from sqlalchemy.sql import text
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from data import conn
from analyze.dump_runs import SELECT

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='predict', help="(predict|export) Use this file to predict via scikitlearn? Or just export cleaned data to csv for R?")
parser.add_argument('-a', '--algo', type=str, default='forest', help="(tree|forest) Which regressor to use (todo add more)")
parser.add_argument('--permute', action="store_true", help="Predict against all permutations of attributes.")
args = parser.parse_args()


def get_clean_data(onehot=True, df=None):
    # If there are recorded runs in the database, use a random forest to find the best combo
    if df is not None:
        dirty = df
    else:
        # sql = 'select hypers, reward_avg from runs where flag is null and array_length(rewards, 1)>250'
        dirty = pd.concat([
            pd.read_sql(SELECT, conn),
            pd.read_csv('runs1.csv', converters={'hypers': json.loads}),
            pd.read_csv('runs2.csv', converters={'hypers': json.loads}),
        ])

        # Expand the hypers dict column to individual columns
        dirty = pd.DataFrame([{**r.hypers, 'reward': r.reward_avg} for r in dirty.itertuples()])
        print(f"{dirty.shape[0]} rows")

    clean = dirty.copy()

    # Impute numerical values if possible
    clean['dropout'].fillna(0., inplace=True)
    clean['gae_lambda'].fillna(1., inplace=True)
    for binary in ['keep_last_timestep', 'indicators', 'scale', 'normalize_rewards']:
        clean[binary] = clean[binary].astype(int)

    if onehot:
        # one-hot encode categorical values
        for cat in [
            'step_optimizer.type',
            'net_type',
            'activation',
            'diff',
            'baseline_mode',
        ]:
            ohc = pd.get_dummies(clean[cat], prefix=cat)
            clean = pd.concat([clean.drop(cat, 1), ohc], axis=1)

    # # Drop columns that only have one value
    for col in clean.columns:
        if len(clean[col].unique()) == 1:
            clean.drop(col, 1, inplace=True)
    # clean.fillna(0, inplace=True)

    # Sort columns by column-name. This way we can be sure when we predict later w/ different data, the columns align.
    clean.sort_index(axis=1, inplace=True)

    return clean, dirty


def predict():
    clean, dirty = get_clean_data()
    X, y = clean.drop('reward', 1), clean['reward']

    # Create model
    tree_hypers = {
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 50, 70]
    }
    if args.algo == 'forest':
        model = RandomForestRegressor()
        hypers = {
            **tree_hypers,
            'n_estimators': [100, 200, 300],  # [10, 20, 50, 100],
            'warm_start': [True, False],
            # 'bootstrap': [True, False],
            'oob_score': [True, False]
        }
    elif args.algo == 'tree':
        model = DecisionTreeRegressor()
        hypers = tree_hypers
    elif args.algo == 'xgboost':
        # TODO (https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/)
        pass

    # Train model w/ cross-val on hypers
    model = GridSearchCV(model, param_grid=hypers, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    model.fit(X.values, y.values)

    # Show scores
    cvres = model.cv_results_
    zipped = sorted(zip(cvres["mean_test_score"], cvres["params"]), key=lambda x: x[0])
    for mean_score, params in zipped:
        print(np.sqrt(-mean_score), params)

    train_pred = model.predict(X)
    print('Train Error', np.sqrt(mean_squared_error(y, train_pred)))

    # https://stackoverflow.com/a/40178961/362790
    tree = model.best_estimator_.estimators_[1] if args.algo == 'forest' else model.best_estimator_

    feature_imp = sorted(zip(model.best_estimator_.feature_importances_, X.columns.values), key=lambda x: x[0], reverse=True)
    print(feature_imp)

    export_graphviz(tree,
                    feature_names=X.columns,
                    filled=True,
                    rounded=True)
    os.system('dot -Tpng tree.dot -o tree.png')

    if args.permute:
        """
        Predict over cross-joined attr/value combos. Doing so in SQL simpler, see https://stackoverflow.com/a/6683506/362790
        SELECT A AS number, B AS letter FROM
        (VALUES (1), (2), (3)) a(A)
        CROSS JOIN
        (VALUES ('A'), ('B'), ('C')) b(B)
        """
        ppo_hypers = hypers_dict['ppo_agent'].copy()

        a, A = 97, 65
        sql_select, sql_body = [], []
        for k, arr in ppo_hypers.items():
            sql_select.append(f'{chr(A)} as "{k}"')
            vals_ = []
            for v in arr:
                vals_.append("NULL" if v is None else f"'{v}'" if type(v) is str else v)
            vals_ = ', '.join([f"({v})" for v in vals_])
            sql_body.append(f"(VALUES {vals_}) {chr(a)}({chr(A)})")
            a+=1;A+=1
        sorted_attrs = [f'"{k}"' for k in sorted(ppo_hypers.keys())]
        sql = "SELECT " + ', '.join(sql_select) \
              + " FROM " + ' CROSS JOIN '.join(sql_body) \
              + " ORDER BY " + ', '.join(sorted_attrs)

        # 201M rows! Find another approach
        offset, buff_size = 0, int(1e4)
        best = None
        while True:
            cartesian = conn.execute(sql + f' LIMIT {buff_size} OFFSET {offset*buff_size}').fetchall()
            if not cartesian: break

            df = pd.DataFrame([{**r, 'reward': 0} for r in cartesian])

            # in order to account for all possible attr values (which effects #one-hot-encoded cols), tag on the
            # training data, then remove it after the xform.
            df = pd.concat([df, dirty])
            X, _ = get_clean_data(df=df)
            X = X.iloc[0:buff_size].drop('reward', 1)

            # print(set(X.columns.values) - set(dirty.columns.values))
            # print(set(dirty.columns.values) - set(X.columns.values))
            pred = model.predict(X)
            pred[pred == 0] = -1e6  # fixme better way to handle zeros? that means the tree's unsure, right?
            max_i = np.argmax(pred)
            if not best or pred[max_i] > best[0]:
                best = (pred[max_i], X.iloc[max_i])
            print('offset', offset, 'reward', best[0])
            print(best[1])
            offset += 1


def export():
    runs = get_clean_data(onehot=False)
    runs.to_csv('runs_out.csv', index=False)


def main():
    if args.mode == 'export':
        export()
        return 0
    predict()


if __name__ == '__main__': main()