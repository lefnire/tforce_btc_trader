import time, json
import pandas as pd
from sqlalchemy import create_engine

config_json = json.load(open('config.json'))
DB = config_json['DB_URL'].split('/')[-1]
engine = create_engine(config_json['DB_URL'])

# From connecting source file, import engine and run the following code. Need each connection to be separate
# (see https://stackoverflow.com/questions/3724900/python-ssl-problem-with-multiprocessing)
# conn = engine.connect()

if DB == 'coins':
    tables = ['okcoin_btccny', 'gdax_btcusd']
    ts_col = 'ts'
    columns = ['last', 'high', 'low', 'volume']
    close_col = 'last'
    predict_col = 'gdax_btcusd_last'
elif DB == 'alex':
    tables = ['exch_ticker_coinbase_usd']
    ts_col = 'trade_timestamp'
    columns = 'last_trade_price last_trade_size bid_price bid_size bid_num_orders ask_price ask_size ask_num_orders'.split(' ')
    close_col = 'last_trade_price'
    predict_col = 'exch_ticker_coinbase_usd_last_trade_price'
elif DB == 'coins2':
    tables = ['g', 'o']
    ts_col = 'close_time'
    columns = ['open', 'high', 'low', 'close', 'volume']
    close_col = 'close'
    predict_col = 'g_close'
elif 'kaggle' in DB:
    tables = ['bitstamp', 'btcn']
    ts_col = 'timestamp'
    columns = ['open', 'high', 'low', 'close', 'volume_btc', 'volume_currency', 'weighted_price']
    close_col = 'close'
    predict_col = 'bitstamp_close'

mode = 'ALL'  # ALL|TRAIN|TEST
def set_mode(m):
    global mode
    mode = m


row_count = 0
train_test_split = 0
already_asked = False
def count_rows(conn):
    global row_count, train_test_split, already_asked
    if row_count:
        return row_count  # cached
    elif already_asked:
        time.sleep(5)  # give the first go a chance to cache a value
    already_asked = True

    row_count = db_to_dataframe(conn, just_count=True)
    train_test_split = int(row_count * .8)
    print('mode: ', mode, ' row_count: ', row_count, ' split: ', train_test_split)
    row_count = train_test_split if mode == 'TRAIN' else row_count - train_test_split
    return row_count


def _db_to_dataframe_ohlc(conn, limit='ALL', offset=0, just_count=False):
    # 600, 300, 1800
    if just_count:
        select = 'select count(*) over () '
        limit, offset = 1, 0
    else:
        select = """
        select 
          g.open_price as g_open, g.high_price as g_high, g.low_price as g_low, g.close_price as g_close, g.volume as g_volume,
          o.open_price as o_open, o.high_price as o_high, o.low_price as o_low, o.close_price as o_close, o.volume as o_volume"""
    query = f"""
    {select}
    from ohlc_gdax as g 
    inner join ohlc_okcoin as o on g.close_time=o.close_time
    where g.period='60' and o.period='60'
    order by g.close_time::integer desc
    limit {limit} offset {offset}
    """
    if just_count:
        return conn.execute(query).fetchone()[0]

    return pd.read_sql_query(query, conn).iloc[::-1].ffill()


def _db_to_dataframe_main(conn, limit='ALL', offset=0, just_count=False):
    """Fetches all relevant data in database and returns as a Pandas dataframe"""
    if just_count:
        query = 'select count(*) over ()'
    else:
        query = 'select ' + ', '.join(
            ', '.join(f"{t}.{c} as {t}_{c}" for c in columns)
            for t in tables
        )

    # interval = 10  # what time-intervals to group by? 60 would be 1-minute intervals
    # TODO https://gis.stackexchange.com/a/127874/105932 for arbitrary interval-grouping
    for (i, table) in enumerate(tables):
        query += " from (" if i == 0 else " inner join ("
        avg_cols = ', '.join(f'avg({c}) as {c}' for c in columns)
        query += f"""
              select {avg_cols},
                date_trunc('second', {ts_col} at time zone 'utc') as ts
              from {table}
              where {ts_col} > now() - interval '2 years'
              group by ts
            ) {table}
            """
        if i != 0:
            query += f'on {table}.ts={tables[i - 1]}.ts'

    if just_count:
        query += " limit 1"
        return conn.execute(query).fetchone()[0]

    query += f" order by {tables[0]}.ts desc limit {limit} offset {offset}"

    # order by date DESC (for limit to cut right), then reverse again (so old->new)
    return pd.read_sql_query(query, conn).iloc[::-1].ffill()


# TODO this just fetches one table because the 3 tables' columns are different (not just in name, but conceptually).
# Need to consolidate them so can do risk-arbitrage
def _db_to_dataframe_alex(conn, limit='ALL', offset=0, just_count=False):
    if just_count:
        return conn.execute('select count(*) from exch_ticker_coinbase_usd').fetchone()[0]

    t = tables[0] # only one for now
    query = 'select ' + ', '.join(f"{c} as {t}_{c}" for c in columns)
    query += f" from {t} order by {ts_col} desc limit {limit} offset {offset}"
    return pd.read_sql_query(query, conn).iloc[::-1].ffill()


def db_to_dataframe(conn, limit='ALL', offset=0, just_count=False):
    global mode, train_test_split
    offset = offset + train_test_split if mode == 'TEST' else offset
    args = [conn, limit, offset, just_count]
    return _db_to_dataframe_alex(*args) if DB == 'alex'\
        else _db_to_dataframe_ohlc(*args) if DB == 'coins2'\
        else _db_to_dataframe_main(*args)


def setup_runs_table(conn):
    conn.execute("""
        --create type if not exists run_flag as enum ('random', 'winner');
        create table if not exists public.runs
        (
            id serial not null,
            hypers jsonb not null,
            reward_avg double precision not null,
            flag varchar(16),
            rewards double precision[],
            agent varchar(64) default 'ppo_agent'::character varying not null,
            actions double precision[],
            prices double precision[]
        );
    """)