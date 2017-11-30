import json
from rl_hsearch import print_feature_importances
from sqlalchemy import create_engine
config_json = json.load(open('config.json'))
import pandas as pd
import numpy as np

engine = create_engine(config_json['OLD_DB_URL'])
conn = engine.connect()
runs = conn.execute("select hypers, reward_avg from runs where flag is null").fetchall()
conn.close()
X, Y = [], []
for run in runs:
    X.append(run.hypers)
    Y.append([run.reward_avg])
X = pd.DataFrame(X)
dummies = pd.get_dummies(X)
dummies.fillna(0, inplace=True)

print_feature_importances(dummies.values, Y, X.columns.values)