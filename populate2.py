import requests, time, json
from sqlalchemy import text
from data import conn


SLEEP = 5*60


def create_table_if_not_exists():
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ohlc(
      ts TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
      exchange VARCHAR(256) NOT NULL,
      data jsonb
    );
    CREATE INDEX IF NOT EXISTS ohlc_ts_idx ON ohlc (ts);
    """)


def fetch_market_and_save():
    """Fetches the most recent market-summaries snapshot and saves to the database. Returns the JSON result from
    the fetch operation. https://cryptowat.ch/docs/api#rate-limit roughly 3s
    """
    #
    try:
        gdax = requests.get('https://api.cryptowat.ch/markets/gdax/btcusd/ohlc').json()['result']
        okcoin = requests.get('https://api.cryptowat.ch/markets/okcoin/btccny/ohlc').json()['result']
    except Exception as e:
        # raise Exception("Cryptowatch allowance out @{}".format(SLEEP))
        print("Cryptowatch allowance out @{}".format(SLEEP))
        return

    query = """
    INSERT INTO ohlc (exchange, data) VALUES ('gdax', :data_gdax);
    INSERT INTO ohlc (exchange, data) VALUES ('okcoin', :data_okcoin);
    """
    conn.execute(text(query), data_gdax=json.dumps(gdax), data_okcoin=json.dumps(okcoin))

i = 0
while True:
    create_table_if_not_exists()
    fetch_market_and_save()
    time.sleep(SLEEP)
    i += 1
    if i % 100 == 0:
        print("ohlc.count: ", conn.execute("select count(*) from ohlc").fetchone().count)