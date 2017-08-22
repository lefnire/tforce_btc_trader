import json
from sqlalchemy import create_engine
from box import Box
import os.path

basepath = os.path.dirname(__file__)
configpath = os.path.abspath(os.path.join(basepath, "config.json"))

config = Box(json.loads(open(configpath).read()))

engine = create_engine(config['data']['db_in']) #, echo=True
conn = engine.connect()

engine_btc = create_engine('postgres://lefnire:lefnire@localhost:5432/btc') #, echo=True
conn_btc = engine_btc.connect()