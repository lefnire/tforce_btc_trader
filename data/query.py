import pandas as pd
from sklearn import preprocessing

from btc import conn, conn_btc, config

btc_data = None

def db_to_dataframe(tail=False):
    """Fetches all relevant data in database and returns as a Pandas dataframe"""
    query = """
    select
      a.ask a_ask, a.volume a_volume,
      b.ask b_ask, b.volume b_volume,
      c.ask c_ask, c.volume c_volume

    from (
      select avg(last) as ask, avg(volume) volume,
        date_trunc('second', ts at time zone 'utc') as ts
      from okcoin_btccny
      where ts > now() - interval '1 year'
      group by ts
    ) a

    inner join (
      select avg(last) as ask, avg(volume) volume,
        date_trunc('second', ts at time zone 'utc') as ts
      from bitstamp_btcusd
      where ts > now() - interval '1 year'
      group by ts
    ) b on a.ts = b.ts

    inner join (
      select avg(last) as ask, avg(volume) volume,
        date_trunc('second', ts at time zone 'utc') as ts
      from gdax_btcusd
      where ts > now() - interval '1 year'
      group by ts
    ) c on b.ts = c.ts

    order by a.ts desc
    """.format('limit 1000' if tail else '')
    df = pd.read_sql_query(query, conn).iloc[::-1]  # order by date DESC (for limit to cut right), then reverse again (so LTR)
    if config.data.sklearn_normalize:
        scaler = preprocessing.MinMaxScaler()  # StandardScaler(copy=True, with_mean=True, with_std=True)
        scaled = pd.DataFrame(scaler.fit_transform(df))
        scaled.columns = df.columns.values
        return scaled
    else:
        return df


def btc_to_dataframe():
    """Fetches all relevant data in database and returns as a Pandas dataframe"""
    global btc_data
    if btc_data:
        return btc_data
    query = """
    select
      a.ask a_ask, a.volume a_volume,
      b.ask b_ask, b.volume b_volume,
      c.ask c_ask, c.volume c_volume

    from (
      select avg(price) ask, avg(volume) volume,
        date_trunc('second', trade_timestamp at time zone 'utc') as ts
      from norm_okcoincny
      --where trade_timestamp > now() - interval '1 year'
      group by ts
    ) a

    inner join (
      select avg(ask) ask, avg(volume) volume,
        date_trunc('second', trade_timestamp at time zone 'utc') as ts
      from norm_bitstampusd
      --where trade_timestamp > now() - interval '1 year'
      group by ts
    ) b on a.ts = b.ts

    inner join (
      select avg(ask) ask, avg(volume) volume,
        date_trunc('second', trade_timestamp at time zone 'utc') as ts
      from norm_coinbaseusd
      --where trade_timestamp > now() - interval '1 year'
      group by ts
    ) c on b.ts = c.ts

    order by a.ts asc
    """
    df = pd.read_sql_query(query, conn_btc)
    if config.data.sklearn_normalize:
        scaler = preprocessing.MinMaxScaler()  # StandardScaler(copy=True, with_mean=True, with_std=True)
        scaled = pd.DataFrame(scaler.fit_transform(df))
        scaled.columns = df.columns.values
        btc_data = scaled
        return scaled
    else:
        btc_data = df
        return df