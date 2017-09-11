import requests, time
from data import conn

SLEEP = 6


def create_table_if_not_exists(tablename):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS {name}(
      id SERIAL PRIMARY KEY,
      last DOUBLE PRECISION,
      high DOUBLE PRECISION,
      low DOUBLE PRECISION,
      change_percent DOUBLE PRECISION,
      change_absolute DOUBLE PRECISION,
      volume DOUBLE PRECISION, 
      ts TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
    );
    CREATE INDEX IF NOT EXISTS {name}_ts_idx ON {name} (ts);
    """.format(name=tablename))


def fetch_market_and_save():
    """Fetches the most recent market-summaries snapshot and saves to the database. Returns the JSON result from
    the fetch operation. https://cryptowat.ch/docs/api#rate-limit roughly 3s
    """
    #
    try:
        res = requests.get('https://api.cryptowat.ch/markets/summaries').json()['result']
    except Exception as e:
        # raise Exception("Cryptowatch allowance out @{}".format(SLEEP))
        print("Cryptowatch allowance out @{}".format(SLEEP))
        return
    query = ""
    for key, val in res.items():
        tablename = key.replace(':', '_').replace('-', '_')
        create_table_if_not_exists(tablename)
        # TODO sanitize via conn.execute(text(query), :a=a, :b=b)
        query += """
        INSERT INTO {name} (last, high, low, change_percent, change_absolute, volume)
        VALUES ({last}, {high}, {low}, {change_percent}, {change_absolute}, {volume});
        """.format(
            name=tablename,
            last=val['price']['last'],
            high=val['price']['high'],
            low=val['price']['low'],
            change_percent=val['price']['change']['percentage'],
            change_absolute=val['price']['change']['absolute'],
            volume=val['volume']
        )
    conn.execute(query)
    return res

i = 0
while True:
    fetch_market_and_save()
    time.sleep(SLEEP)
    i += 1
    if i % 100 == 0:
        print("gdax_btcusd.count: ", conn.execute("select count(*) from gdax_btcusd").fetchone().count)