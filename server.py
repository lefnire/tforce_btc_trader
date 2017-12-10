# Run as $ FLASK_APP=server.py flask run
import json, pdb, pprint
from flask import Flask, jsonify
import data
from flask_cors import CORS
from sqlalchemy import create_engine
import pandas as pd
from rl_hsearch import print_feature_importances

app = Flask(__name__)
CORS(app)

db_url = data.config_json['DB_URL']
databases = dict(
    # kaggle=create_engine(db_url.replace(data.DB, 'kaggle')),
    # kaggle2=create_engine(db_url.replace(data.DB, 'kaggle2')),
    # alex=create_engine(db_url.replace(data.DB, 'alex'))
    alex2=create_engine(db_url.replace(data.DB, 'alex2'))
)

@app.route("/")
def send_data():
    rows = []
    X, Y = [], []
    for db_name, engine in databases.items():
        conn = engine.connect()
        for row in conn.execute('select * from runs').fetchall():
            row = dict(row.items())
            row['source'] = db_name
            rows.append(row)
            X.append({'source': row['source'], **row['hypers']})
            Y.append([row['reward_avg']])
        conn.close()

    X = pd.get_dummies(pd.DataFrame(X))
    X.fillna(0., inplace=True)
    print('#rows:', len(X))
    print_feature_importances(X, Y, X.columns.values)

    return jsonify(rows)
