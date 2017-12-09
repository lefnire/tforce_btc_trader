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
elif 'alex' in DB:
    tables = [
    {
        'name': 'exch_ticker_kraken_usd',
        'ts_col': 'last_update',
        'columns': 'last_trade_price last_trade_lot_volume ask_price ask_lot_volume bid_price bid_lot_volume volume volume_last24 vwap vwap_last24 number_trades number_trades_last24 low low_last24 high high_last24 open_price'.split(' '),
        'ohlcv': {'open': 'open_price', 'high': 'high', 'low': 'low', 'close': 'last_trade_price', 'volume': 'volume'},
    },
    {
        'name': 'exch_ticker_coinbase_usd',
        'ts_col': 'last_update',
        'columns': 'last_trade_price last_trade_size bid_price bid_size bid_num_orders ask_price ask_size ask_num_orders'.split(' '),
    }
    ]
    target = 'exch_ticker_kraken_usd_last_trade_price'
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

def get_tables(arbitrage=True):
    return tables if arbitrage else [tables[0]]

def n_cols(conv2d=False, indicators=False, arbitrage=True):
    cols = 0
    tables_ = get_tables(arbitrage)
    for t in tables_:
        cols += len(t['columns'])
        if indicators and 'ohlcv' in t:
            cols += 4  # This is defined in btc_env._get_indicators()
    if not conv2d:
        cols += 2  # [self.cash, self.value] are added in downstream dense
    return cols

mode = 'ALL'  # ALL|TRAIN|TEST
def set_mode(m):
    global mode
    mode = m


row_count = 0
train_test_split = 0
already_asked = False
def count_rows(conn, arbitrage=True):
    global row_count, train_test_split, already_asked
    if row_count:
        return row_count  # cached
    elif already_asked:
        time.sleep(5)  # give the first go a chance to cache a value
    already_asked = True

    row_count = db_to_dataframe(conn, just_count=True, arbitrage=arbitrage)
    if mode == 'ALL':
        print('mode: ', mode, ' row_count: ', row_count)
        return row_count
    train_test_split = int(row_count * .8)
    print('mode: ', mode, ' row_count: ', row_count, ' split: ', train_test_split)
    row_count = train_test_split if mode == 'TRAIN' else row_count - train_test_split
    return row_count


def _get_offset(offset):
    global mode, train_test_split
    return offset + train_test_split if mode == 'TEST' else offset

def _db_to_dataframe_ohlc(conn, limit='ALL', offset=0, just_count=False, arbitrage=True):
    # 600, 300, 1800
    offset = _get_offset(offset)

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


def _db_to_dataframe_main(conn, limit='ALL', offset=0, just_count=False, arbitrage=True):
    """Fetches all relevant data in database and returns as a Pandas dataframe"""
    offset = _get_offset(offset)
    tables_ = get_tables(arbitrage)

    if just_count:
        query = 'select count(*) over ()'
    else:
        query = 'select ' + ', '.join(
            ', '.join(f"{t['name']}.{c} as {t['name']}_{c}" for c in t['columns'])
            for t in tables_
        )

    # Currently matching main-table to nearest secondary-tables' time (https://stackoverflow.com/questions/28839524/join-two-tables-based-on-nearby-timestamps)
    # Could also group by intervals (10s, 60s, etc) https://gis.stackexchange.com/a/127874/105932
    first = tables_[0]
    for i, table in enumerate(tables_):
        name, ts = table['name'], table['ts_col']
        if i == 0:
            query += f" from {name}"
            continue
        prior = tables_[i-1]
        query += f"""
            left join lateral (
              select {', '.join(c for c in table['columns'])}
              from {name}
              where {name}.{ts} <= {prior['name']}.{prior['ts_col']}
              order by {name}.{ts} desc
              limit 1 
            ) {name} on true
            """

    if just_count:
        query += " limit 1"
        return conn.execute(query).fetchone()[0]

    order_field = f"{first['name']}.{first['ts_col']}" if len(tables_) > 1 else first['ts_col']
    query += f" order by {order_field} desc limit {limit} offset {offset}"

    # order by date DESC (for limit to cut right), then reverse again (so old->new)
    df = pd.read_sql_query(query, conn).iloc[::-1]
    # return df.fillna(method='ffill').fillna(method='bfill')  # fill forward then backward
    return df

db_to_dataframe = _db_to_dataframe_ohlc if DB == 'coins2'\
    else _db_to_dataframe_main


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