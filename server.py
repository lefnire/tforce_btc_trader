# Run as $ FLASK_APP=server.py flask run
import json, pdb, pprint
from flask import Flask, jsonify
import data
from flask_cors import CORS
from sqlalchemy import create_engine
import utils

app = Flask(__name__)
CORS(app)

db_url = data.config_json['DB_URL']
engine = create_engine(db_url)

@app.route("/")
def send_data():
    rows = []
    conn = engine.connect()
    # TODO prices/actions in separate route
    for row in conn.execute('select id, hypers, advantage_avg, advantages, uniques from runs').fetchall():
        row = dict(row.items())
        row['advantage_avg'] = utils.calculate_score(row)
        rows.append(row)
    conn.close()

    print(len(rows), 'rows')
    return jsonify(rows)
