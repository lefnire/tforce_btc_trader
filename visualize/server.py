# Run as $ FLASK_APP=server.py flask run
import json, pdb, pprint
from flask import Flask, jsonify
from data.data import engine_runs
from flask_cors import CORS
from sqlalchemy import create_engine, text
import utils

app = Flask(__name__)
CORS(app)


@app.route("/")
def get_runs():
    rows = []
    conn = engine_runs.connect()
    # TODO prices/actions in separate route
    for row in conn.execute('select id, hypers, advantage_avg, advantages, uniques from runs where advantage_avg>0').fetchall():
        row = dict(row.items())
        row['advantage_avg'] = utils.calculate_score(row['advantages'])
        rows.append(row)
    conn.close()

    print(len(rows), 'rows')
    return jsonify(rows)


@app.route("/actions/<run_id>")
def get_actions(run_id):
    conn = engine_runs.connect()
    query = 'select actions, prices from runs where id=:run_id'
    row = conn.execute(text(query), run_id=run_id).fetchone()
    conn.close()

    return jsonify(dict(row))

