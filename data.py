import json, os.path
import pandas as pd
from sklearn import preprocessing
from sqlalchemy import create_engine
from box import Box

basepath = os.path.dirname(__file__)
configpath = os.path.abspath(os.path.join(basepath, "config.json"))
config = Box(json.loads(open(configpath).read()))


engine = create_engine(config.data.db_in)
conn = engine.connect()

source = 'btc' if config.data.db_in.endswith('btc') else 'coins'
tables = ['norm_btcncny', 'norm_bitstampusd', 'norm_coinbaseusd'] if source == 'btc'\
    else ['okcoin_btccny', 'bitstamp_btcusd', 'gdax_btcusd']
columns = ['last', 'high', 'low', 'volume']
ts_col = 'ts' if source == 'coins' else 'trade_timestamp'


def db_to_dataframe(limit=None):
    """Fetches all relevant data in database and returns as a Pandas dataframe"""
    # TODO cols we should use: high, low, volume(check) OPEN, CLOSE

    query = 'select ' + ', '.join(
        ', '.join('{t}.{c} as {t}_{c}'.format(t=t, c=c) for c in columns)
        for t in tables
    )

    for (i, table) in enumerate(tables):
        query += " from (" if i == 0 else " inner join ("
        avg_cols = ', '.join('avg({c}) as {c}'.format(c=c) for c in columns)
        query += """
          select {avg_cols},
            date_trunc('second', {ts_col} at time zone 'utc') as ts
          from {table}
          where {ts_col} > now() - interval '1 year'
          group by ts
        ) {table}
        """.format(table=table, ts_col=ts_col, avg_cols=avg_cols)
        if i != 0:
            query += 'on {a}.ts={b}.ts'.format(a=table, b=tables[i-1])

    query += " order by {t}.ts desc".format(t=tables[0])
    if limit:
        query += ' limit {}'.format(limit)

    # print(query)
    df = pd.read_sql_query(query, conn).iloc[::-1]  # order by date DESC (for limit to cut right), then reverse again (so LTR)
    if config.data.sklearn_normalize:
        scaler = preprocessing.MinMaxScaler()  # StandardScaler(copy=True, with_mean=True, with_std=True)
        scaled = pd.DataFrame(scaler.fit_transform(df))
        scaled.columns = df.columns.values
        return scaled
    else:
        return df