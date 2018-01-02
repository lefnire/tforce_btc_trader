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
    conn = engine.connect()
    for row in conn.execute('select * from runs').fetchall():
        row = dict(row.items())
        rows.append(row)
    conn.close()

    print(len(rows), 'rows')
    return jsonify(rows)
