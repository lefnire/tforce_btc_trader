# import tensorflow  # this has to be imported _before_ pandas even if not used
from hypersearch import hypers as hypers_dict # actually, handle the ordering in hypersearch.py & import that here
from functools import reduce
import pandas as pd
import os, pdb, json, argparse
import numpy as np
from sqlalchemy.sql import text
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from data import conn

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', type=str, default='ppo_agent', help="(ppo_agent|dqn_agent) agent to use")
parser.add_argument('-m', '--model', type=str, default='boost', help="(tree|forest|boost|svr) Which regressor to use (todo add more)")
parser.add_argument('-r', '--reward-mode', type=str, default='avg', help="(avg|gt0) Should we sort runs by last-50 reward-averages? Or by the number of net-gain episodes (> 0).")
parser.add_argument('--predict', action="store_true", help="Predict against all permutations of attributes.")
args = parser.parse_args()


def parse_rewards_json(r):
    r = r.replace('{','[').replace('}',']')
    return json.loads(r)


def get_clean_data(onehot=True, df=None):
    # If there are recorded runs in the database, use a random forest to find the best combo
    if df is not None:
        dirty = df
    else:
        # sql = 'select hypers, reward_avg from runs where flag is null and array_length(rewards, 1)>250'
        dirty = pd.concat([
            pd.read_sql('select hypers, reward_avg, rewards, agent from runs', conn),
            pd.read_csv('runs1.csv', converters={'hypers': json.loads, 'rewards': parse_rewards_json}),
            pd.read_csv('runs2.csv', converters={'hypers': json.loads, 'rewards': parse_rewards_json}),
            pd.read_csv('runs3.csv', converters={'hypers': json.loads, 'rewards': parse_rewards_json}),
        ])

        # dirty = dirty[dirty.agent == 'ppo_agent']

        # Expand the hypers dict to individual columns. Use more flexible avg
        reward_avg = pd.DataFrame([
            {**d.hypers, 'reward': np.mean(d.rewards[-20:])}
            for d in dirty.itertuples()
            if len(d.rewards) > 150
        ])
        gt0 = pd.DataFrame([
            {**d.hypers, 'reward': (np.array(d.rewards[-20:]) > 0).sum()}
            for d in dirty.itertuples()
            if len(d.rewards) > 150
        ])
        dirty = reward_avg if args.reward_mode == 'avg' else gt0

        print(f"{dirty.shape[0]} rows")
        top_avg = reward_avg.sort_values('reward', ascending=False).iloc[:10]
        top_gt0 = gt0.sort_values('reward', ascending=False).iloc[:10]
        f = open('out.txt', 'w')

        f.write('# Top')
        f.write('\n## Avg\n')
        f.write(top_avg.iloc[0].to_string())
        f.write('\n## > 0\n')
        f.write(top_gt0.iloc[0].to_string())

        f.write('\n\n# Top[:10]')
        f.write('\n## Avg\n')
        f.write(top_avg.to_string())
        f.write('\n## > 0\n')
        f.write(top_gt0.to_string())

        f.write('\n\n# Mode')
        f.write('\n## Avg\n')
        f.write(top_avg.mode().iloc[0:2].fillna('-').to_string())
        f.write('\n## >0\n')
        f.write(top_gt0.mode().iloc[0:2].fillna('-').to_string())
        f.close()

        df_avg = top_avg if args.reward_mode == 'avg' else top_gt0
        mode_json = df_avg.drop('reward',1).mode().iloc[0].to_json()
        conn.execute("delete from runs where flag='mode'")
        conn.execute(text(
            "insert into runs (hypers, reward_avg, rewards, flag) values (:hypers, :reward_avg, :rewards, 'mode')"),
                     hypers=mode_json, reward_avg=np.mean(df_avg['reward']), rewards=[])

    for possible_missing_col in ['cryptowatch']:
        if possible_missing_col not in dirty:
            missing_col = pd.DataFrame(np.expand_dims(np.zeros(dirty.shape[0]), 1), columns=[possible_missing_col])
            dirty = pd.concat([dirty, missing_col], 1)

    clean = dirty.copy()

    # Impute numerical values if possible
    for to_false in ['indicators', 'scale']:
        clean[to_false].fillna(False, inplace=True)
    for to_0 in ['dropout', 'gae_lambda', 'penalize_inaction', 'cryptowatch']:
        clean[to_0].fillna(0., inplace=True)
    for binary in ['keep_last_timestep', 'indicators', 'scale', 'normalize_rewards', 'cryptowatch', 'penalize_inaction']:
        try:
            clean[binary] = clean[binary].astype(int)
        except Exception as e:
            pdb.set_trace()

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
    clean.fillna(0, inplace=True)

    # Sort columns by column-name. This way we can be sure when we predict later w/ different data, the columns align.
    clean.sort_index(axis=1, inplace=True)

    return clean, dirty


def predict():
    clean, dirty = get_clean_data()
    X, y = clean.drop('reward', 1), clean['reward']

    # Create model
    tree_hypers = {
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [None, 10, 20]
    }
    if args.model == 'forest':
        model = RandomForestRegressor(bootstrap=True, oob_score=True)
        hypers = {
            **tree_hypers,
            'n_estimators': [100, 200, 300],
        }
    elif args.model == 'tree':
        model = DecisionTreeRegressor()
        hypers = tree_hypers
    elif args.model == 'boost':
        model = GradientBoostingRegressor()
        hypers = {
            **tree_hypers,
            'n_estimators': [100, 200, 300],
        }
    elif args.model == 'xgboost':
        # https://jessesw.com/XG-Boost/
        import xgboost as xgb
        model = xgb.XGBRegressor()
        hypers = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
    elif args.model == 'svr':
        model = SVR()
        hypers = {'C': [1, 10, 15]}
    elif args.model == 'dnn':
        model = MLPRegressor()
        hypers = {
            'hidden_layer_sizes': [(32, 16, 8), (64, 32, 16), (64, 32, 16, 8)],
            'max_iter': [800, 2000]
        }


    # Train model w/ cross-val on hypers
    if args.predict:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    model = GridSearchCV(model, param_grid=hypers, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model.fit(X_train, y_train)

    # Show scores
    cvres = model.cv_results_
    zipped = sorted(zip(cvres["mean_test_score"], cvres["params"]), key=lambda x: x[0])
    for mean_score, params in zipped:
        print(np.sqrt(-mean_score), params)

    test_pred = model.predict(X_test)
    print('Test Error', np.sqrt(mean_squared_error(y_test, test_pred)))

    if args.model == 'forest':
        print('OOB Score', model.best_estimator_.oob_score_)


    if args.model in ('forest', 'tree'):
        feature_imp = sorted(zip(model.best_estimator_.feature_importances_, X.columns.values), key=lambda x: x[0], reverse=True)
        f = open('out.txt', 'a')
        f.write('\n\n--- Feature Importances ---\n')
        f.write(', '.join([f'{x[1]}: {round(x[0],4)}' for x in feature_imp]))
        f.close()

        # https://stackoverflow.com/a/40178961/362790
        tree = model.best_estimator_.estimators_[1] if args.model == 'forest' else model.best_estimator_
        export_graphviz(tree,
                        feature_names=X.columns,
                        filled=True,
                        rounded=True)
        os.system('dot -Tpng tree.dot -o tree.png;rm tree.dot')

    if args.predict:
        """
        Predict over cross-joined attr/value combos. Doing so in SQL simpler, see https://stackoverflow.com/a/6683506/362790
        SELECT A AS number, B AS letter FROM
        (VALUES (1), (2), (3)) a(A)
        CROSS JOIN
        (VALUES ('A'), ('B'), ('C')) b(B)
        """
        hypers = hypers_dict[args.agent].copy()
        # Get rid of dependencies structure (flatten)
        for k, v in hypers.items():
            if type(v) is dict:
                hypers[k] = v['$vals']
        hypers.update(hypers_dict['custom'])

        a, A = 97, 65
        sql_select, sql_body = [], []
        for k, arr in hypers.items():
            sql_select.append(f'{chr(A)} as "{k}"')
            vals_ = []
            for v in arr:
                vals_.append("NULL" if v is None else f"'{v}'" if type(v) is str else v)
            vals_ = ', '.join([f"({v})" for v in vals_])
            sql_body.append(f"(VALUES {vals_}) {chr(a)}({chr(A)})")
            a+=1;A+=1
        sql = "SELECT " + ', '.join(sql_select) \
              + " FROM " + ' CROSS JOIN '.join(sql_body)

        cartesian_size = reduce((lambda m, vals: len(vals) * m), hypers.values(), 1)
        # stream_results is key - lets us fetchmany() w/o storing all results in mem, and w/o requiring LIMIT/OFFSET + ORDER BY
        cartesian = conn.execution_options(stream_results=True).execute(sql)
        buff_size, i = int(1e5), 1
        best = {'reward': -1e6, 'flat': None}
        while True:
            chunk = cartesian.fetchmany(buff_size)
            if i % 10 == 0:
                print(str(round(i*buff_size/cartesian_size*100, 2)) + '%')
            if not chunk: break

            df = pd.DataFrame([dict(r) for r in chunk])

            # in order to account for all possible attr values (which effects #one-hot-encoded cols), tag on the
            # training data, then remove it after the xform.
            df = pd.concat([df, dirty.drop('reward',1)])
            X_test, _ = get_clean_data(df=df)
            X_test = X_test.iloc[0:buff_size]

            # print(set(X_test.columns.values) - set(X.columns.values))
            # print(set(X.columns.values) - set(X_test.columns.values))
            pred = model.predict(X_test)
            pred[pred == 0] = -1e6  # fixme better way to handle zeros? that means the tree's unsure, right?
            max_i = np.argmax(pred)
            if pred[max_i] > best['reward']:
                best['flat'], best['reward'] = df.iloc[max_i], pred[max_i]
                print()
                print('Reward', best['reward'])
                print(best['flat'])
            i += 1
        f = open('out.txt', 'a')
        f.write('\n\n# Top Predicted\n')
        f.write(best['flat'].to_string())
        f.close()

        conn.execute("delete from runs where flag='predicted'")
        conn.execute(text("insert into runs (hypers, reward_avg, rewards, flag) values (:hypers, :reward_avg, :rewards, 'predicted')"),
                     hypers=best['flat'].to_json(), reward_avg=best['reward'], rewards=[])


def main():
    predict()


if __name__ == '__main__': main()