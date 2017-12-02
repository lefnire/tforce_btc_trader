# Run as $ FLASK_APP=server.py flask run
import json, pdb, pprint
from flask import Flask, jsonify
import data
from flask_cors import CORS
from sqlalchemy import create_engine

app = Flask(__name__)
CORS(app)

db_url = data.config_json['DB_URL']
databases = dict(
    kaggle=create_engine(db_url.replace(data.DB, 'kaggle')),
    kaggle2=create_engine(db_url.replace(data.DB, 'kaggle2')),
    alex=create_engine(db_url.replace(data.DB, 'alex'))
)

@app.route("/")
def send_data():
    rows = []
    for db_name, engine in databases.items():
        conn = engine.connect()
        for row in conn.execute('select * from runs').fetchall():
            row = dict(row.items())
            row['source'] = db_name
            rows.append(row)
        conn.close()
    return jsonify(rows)