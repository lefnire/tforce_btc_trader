import json
from sqlalchemy.sql import text
from data import engine

conn = engine.connect()
rows = conn.execute('select id,hypers from runs').fetchall()
for i, r in enumerate(rows):
    h = r['hypers']
    h['depth_pre'] = h['pre_depth']
    h['depth_mid'] = h['depth']
    h['depth_post'] = h['depth']
    del h['pre_depth']
    del h['depth']
    print(i, h)
    # conn.execute(text('update runs set hypers=:h where id=:id'), h=json.dumps(h), id=r['id'])