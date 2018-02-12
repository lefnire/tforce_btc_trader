import json
from sqlalchemy.sql import text
from data import engine_runs

conn = engine_runs.connect()
rows = conn.execute('select id, hypers from runs').fetchall()
for i, r in enumerate(rows):
    h = r['hypers']
    h['reward_type'] = 'raw'
    if h['advantage_reward']:
        h['reward_type'] = 'advantage'
    del h['advantage_reward']
    # print(i, h)
    # print()
    conn.execute(text('update runs set hypers=:h where id=:id'), h=json.dumps(h), id=r['id'])