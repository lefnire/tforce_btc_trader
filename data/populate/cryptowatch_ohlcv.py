import requests, time, json
from sqlalchemy import text
from sqlalchemy import create_engine

conn = create_engine('postgres://lefnire:lefnire@localhost/cryptowatch').connect()


SLEEP = 10*60  # can probably be 500*60


def create_table_if_not_exists(name):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ohlc_{n} (
      close_time INTEGER NOT NULL, -- TIMESTAMP NOT NULL,
      period VARCHAR(16) NOT NULL, -- "60" for 1min candles, "180" for 3m, etc 
      
      open_price DOUBLE PRECISION,
      high_price DOUBLE PRECISION,
      low_price DOUBLE PRECISION,
      close_price DOUBLE PRECISION,
      volume DOUBLE PRECISION,
      
      PRIMARY KEY (close_time, period)
    );
    """.format(n=name))


def fetch_market_and_save():
    """https://cryptowat.ch/docs/api#ohlc"""
    try:
        res = dict(
            gdax=requests.get('https://api.cryptowat.ch/markets/gdax/btcusd/ohlc').json()['result'],
            okcoin=requests.get('https://api.cryptowat.ch/markets/okcoin/btccny/ohlc').json()['result']
        )
    except Exception as e:
        # raise Exception("Cryptowatch allowance out @{}".format(SLEEP))
        print("Cryptowatch allowance out @{}".format(SLEEP))
        return

    for exchange, periods in res.items():
        for period, candles in periods.items():
            for candle in candles:
                query = """
                INSERT INTO ohlc_{n} (close_time, period, open_price, high_price, low_price, close_price, volume) 
                VALUES (:close_time, :period, :open_price, :high_price, :low_price, :close_price, :volume)
                ON CONFLICT DO NOTHING; -- there _will_ be overlap periods each iteration, which don't change
                """.format(n=exchange)
                conn.execute(text(query),
                             period=period,
                             close_time=candle[0],
                             open_price=candle[1],
                             high_price=candle[2],
                             low_price=candle[3],
                             close_price=candle[4],
                             volume=candle[5]
                )

i = 0
while True:
    create_table_if_not_exists('gdax')
    create_table_if_not_exists('okcoin')
    fetch_market_and_save()
    print("ohlc.count: ", conn.execute("select count(*) from ohlc_gdax where period='60'").fetchone().count)
    time.sleep(SLEEP)
    i += 1
