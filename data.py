import pandas as pd
from sqlalchemy import create_engine

DB = "coins2"

engine = create_engine("postgres://lefnire:lefnire@localhost:5432/{}".format(DB))
conn = engine.connect()

if DB == 'coins':
    tables = ['okcoin_btccny', 'gdax_btcusd']
    ts_col = 'ts'
    columns = ['last', 'high', 'low', 'volume']
elif DB.endswith('btc'):  # alex's database
    tables = ['norm_btcncny', 'norm_bitstampusd', 'norm_coinbaseusd']
    ts_col = 'trade_timestamp'
    columns = ['last', 'high', 'low', 'volume']
elif DB == 'coins2':
    tables = ['g', 'o']
    ts_col = 'close_time'
    columns = ['open', 'high', 'low', 'close', 'volume']


def count_rows():
    # FIXME! when using data other than Tyler's, need to use count based on db_to_dataframe query, which will count
    # the inner-joined (many rows will disappear and this query won't work)
    if DB == 'coins2':
        return conn.execute("""
            select count(*)
            from ohlc_gdax g
            inner join ohlc_okcoin o on o.close_time=g.close_time
            where g.period='60' and o.period='60';
        """).fetchone()[0]
    return conn.execute('select count(*) from {}'.format(tables[0])).fetchone()[0]


def _db_to_dataframe_ohlc(limit='ALL', offset=0):
    # 600, 300, 1800
    query = """
    select 
      g.open_price as g_open, g.high_price as g_high, g.low_price as g_low, g.close_price as g_close, g.volume as g_volume,
      o.open_price as o_open, o.high_price as o_high, o.low_price as o_low, o.close_price as o_close, o.volume as o_volume
    from ohlc_gdax as g 
    inner join ohlc_okcoin as o on g.close_time=o.close_time
    where g.period='60' and o.period='60'
    order by g.close_time::integer desc
    limit {limit} offset {offset}
    """.format(limit=limit, offset=offset)
    return pd.read_sql_query(query, conn).iloc[::-1].ffill()

def _db_to_dataframe_main(limit='ALL', offset=0):
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
    return pd.read_sql_query(query, conn).iloc[::-1].ffill()

def db_to_dataframe(limit='ALL', offset=0):
    return _db_to_dataframe_ohlc(limit=limit, offset=offset) if DB == 'coins2'\
        else _db_to_dataframe_main(limit=limit, offset=offset)