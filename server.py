# Run as $ FLASK_APP=server.py flask run
import json, pdb, pprint
from flask import Flask, jsonify
import data
from flask_cors import CORS
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from rl_hsearch import print_feature_importances

app = Flask(__name__)
CORS(app)

db_url = data.config_json['DB_URL']
engine = create_engine(db_url)

@app.route("/")
def send_data():
    rows = []
    X, Y = [], []
    conn = engine.connect()
    for row in conn.execute('select * from runs').fetchall():
        row = dict(row.items())
        rows.append(row)
        X.append(row['hypers'])
        row['reward_avg'] = (row['reward_avg_human'] + row['reward_avg_agent']) / 2
        Y.append([row['reward_avg']])
    conn.close()

    X = pd.get_dummies(pd.DataFrame(X))
    X.fillna(0., inplace=True)
    print('#rows:', len(X))
    print_feature_importances(X, Y, X.columns.values)

    return jsonify(rows)
