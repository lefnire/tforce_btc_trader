import pandas as pd
from sqlalchemy import create_engine

DB = "coins"

engine = create_engine("postgres://lefnire:lefnire@localhost:5432/{}".format(DB))
conn = engine.connect()

source = 'btc' if DB.endswith('btc') else 'coins'
tables = ['norm_btcncny', 'norm_bitstampusd', 'norm_coinbaseusd'] if source == 'btc'\
    else ['okcoin_btccny', 'gdax_btcusd']
    # else ['okcoin_btccny', 'bitstamp_btcusd', 'gdax_btcusd']
columns = ['last', 'high', 'low', 'volume']
ts_col = 'ts' if source == 'coins' else 'trade_timestamp'


def count_rows():
    # FIXME! when using data other than Tyler's, need to use count based on db_to_dataframe query, which will count
    # the inner-joined (many rows will disappear and this query won't work)
    return conn.execute('select count(*) from {}'.format(tables[0])).fetchone()[0]


def db_to_dataframe(limit='ALL', offset=0, scaler=None, scaler_args={}):
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

    query += " order by {}.ts desc limit {} offset {}".format(tables[0], limit, offset)

    # order by date DESC (for limit to cut right), then reverse again (so old->new)
    df = pd.read_sql_query(query, conn).iloc[::-1].ffill()
    if scaler:
        scaler = scaler(**scaler_args)  # StandardScaler(copy=True, with_mean=True, with_std=True)
        scaled = pd.DataFrame(scaler.fit_transform(df))
        scaled.columns = df.columns.values
        df = scaled
    return df