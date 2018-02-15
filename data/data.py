import time, json, re
from enum import Enum
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import os

# From connecting source file, `import engine` and run `engine.connect()`. Need each connection to be separate
# (see https://stackoverflow.com/questions/3724900/python-ssl-problem-with-multiprocessing)
config_json = json.load(open(os.path.dirname(__file__) + '/../config.json'))
DB = config_json['DB_HISTORY'].split('/')[-1]
engine = create_engine(config_json['DB_HISTORY'])
engine_live = create_engine(config_json['DB_HISTORY_LIVE'])
engine_runs = create_engine(config_json['DB_RUNS'])



# Decide which exchange you want to trade on (significant even in training). Pros & cons; Kraken's API provides more
# details than GDAX (bid/ask spread, VWAP, etc) which means predicting its next price-action is easier for RL. It
# also has a lower minimum trade (.002 BTC vs GDAX's .01 BTC), which gives it more wiggle room. However, its API is
# very unstable and slow, so when you actually go live you'r bot will be suffering. GDAX's API is rock-solid. Look
# into the API stability, it may change by the time you're using this. If Kraken is solid, use it instead.
class Exchange(Enum):
    GDAX = 'gdax'
    KRAKEN = 'kraken'

EXCHANGE = Exchange.KRAKEN

# Methods for imputing NaN. F=ffill, B=bfill, Z=zero. Generally we want volume/size-based features to be 0-filled
# (indicating no trading during this blank period) and prices to ffill (maintain where the price left off). Right?
F = 0
B = 1
Z = 2


# Y'all won't have access to this database, this is a friend's personal DB. Don't worry, the Kaggle dataset is
# fuller & cleaner, this is just one that's live/real-time which is why I use it. If anyone knows another (even
# if paid) LMK.
if 'alex' in DB or DB == 'dbk0cfbk3mfsb6':
    tables = [
    {
        'name': 'exch_ticker_coinbase_usd',
        'ts': 'last_update',
        'cols': dict(
            last_trade_price=F,
            last_trade_size=Z,
            bid_price=F,
            bid_size=Z,
            bid_num_orders=Z,
            ask_price=F,
            ask_size=Z,
            ask_num_orders=Z
        )
    },
    {
        'name': 'exch_ticker_kraken_usd',
        'ts': 'last_update',
        'cols': dict(
            last_trade_price=F,
            last_trade_lot_volume=Z,
            ask_price=F,
            ask_lot_volume=Z,
            bid_price=F,
            bid_lot_volume=Z,
            volume=Z,
            #volume_last24=Z,
            vwap=Z,
            #vwap_last24=Z,
            number_trades=Z,
            #number_trades_last24=Z,
            low=F,
            #low_last24=F,
            high=F,
            #high_last24=F,
            open_price=F
        ),
        'ohlcv': dict(
            open='open_price',
            high='high',
            low='low',
            close='last_trade_price',
            volume='volume'
        ),
    },
    ]
    target = 'exch_ticker_coinbase_usd_last_trade_price'

    # order of tables counts - first one should be the table containing the target
    if EXCHANGE == Exchange.KRAKEN:
        tables[0], tables[1] = tables[1], tables[0]
        target = 'exch_ticker_kraken_usd_last_trade_price'

else:
    tables = [
    {
        'name': 'coinbase',
        'ts': 'timestamp',
        'cols': dict(
            open=F,
            high=F,
            low=F,
            close=F,
            volume_btc=Z,
            volume_currency=Z,
            weighted_price=Z
        ),
        'ohlcv': dict(
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume_currency'
        )
    },
    {
        'name': 'coincheck',
        'ts': 'timestamp',
        'cols': dict(
            open=F,
            high=F,
            low=F,
            close=F,
            volume_btc=Z,
            volume_currency=Z,
            weighted_price=Z
        ),
        'ohlcv': dict(
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume_currency'
        )
    }
    ]
    target = 'coinbase_close'


def get_tables(arbitrage=True):
    return tables if arbitrage else [tables[0]]


def n_cols(indicators=False, arbitrage=True):
    cols = 0
    tables_ = get_tables(arbitrage)
    for t in tables_:
        cols += len(t['cols'])
        if indicators and 'ohlcv' in t:
            cols += 4  # This is defined in btc_env._get_indicators()
    # Extra 3 cols (self.cash, self.value, self.repeats) are added in downstream dense
    return cols


row_count = 0
already_asked = False
def count_rows(conn, arbitrage=True):
    global row_count, already_asked

    # This fn might be called suddenly a bunch in parallel - try to let one instance fetch the count first & cache
    if row_count:
        return row_count  # cached
    elif already_asked:
        time.sleep(5)  # give the first go a chance to cache a value
    already_asked = True

    row_count = db_to_dataframe(conn, just_count=True, arbitrage=arbitrage)
    print('row_count: ', row_count)
    return row_count


def _db_to_dataframe_ohlc(conn, limit='ALL', offset=0, just_count=False, arbitrage=True):
    """This fn is currently not used anywhere. You'd use this if using the CryptoWat.ch OHLCV data (see
    data/populate/cryptowatch_ohlcv.py). Fantastic dataset, with hierarchical candlesticks! But not enough history to
    train on. I hope they sell full history some day.
    """
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


def _db_to_dataframe_main(conn, limit='ALL', offset=0, just_count=False, arbitrage=True, last_timestamp=False):
    """
    Fetches data from your `history` database. During training, this'll fetch 80% of the data (TODO: buffer that
    instead so it's not so RAM-heavy). During testing, 20% unseen data.
    :param conn: a database connection
    :param limit: num rows to fetch
    :param offset: n-rows to start from. Note! This function fetches from newest-to-oldest, so offset=0 means
        most-recent. The function reverses that in the end so we're properly sequential. I don't remember why I did
        this.. maybe makes limit/offset easier since I don't need to track database end? Perhaps this can change.
    :param just_count: True if you just want to count the rows (used up-front in btc_env to set some internals).
        You may be thinking "just do a `select count(*)`, why fn(just_count=True)? Because the `arbitrage` arg may
        change the resultant row-count, see below.
    :param arbitrage: Whether to use "risk arbitrage" (effects the row count due to time-phase alignment). See
        hypersearch.py for info on this arg.
    :param last_timestamp: When we're in live-mode, we run till the last row in our database, use this arg to track
        where we left off, wait, poll if new rows, repeat.
    :return: pd.DataFrame, with NaNs imputed according to the F/B/Z rules
    """
    tables_ = get_tables(arbitrage)

    if just_count:
        query = 'select count(*) over ()'
    else:
        query = 'select ' + ', '.join(
            ', '.join(f"{t['name']}.{c} as {t['name']}_{c}" for c in t['cols'])
            for t in tables_
        )

    # Currently matching main-table to nearest secondary-tables' time (https://stackoverflow.com/questions/28839524/join-two-tables-based-on-nearby-timestamps)
    # The current method is an OUTER JOIN, which means all primary-table's rows are kept, and any most-recent
    # secondary-tables' rows are matched - no matter how "recent" they are. Could cause problems, what if the
    # most-recent match is 1 day ago? The alternative is an INNER JOIN via a hard-coded time interval (GROUP BY 10s,
    # 60s, etc - https://gis.stackexchange.com/a/127874/105932). With that approach you lose rows that don't have a
    # match, and therefore get "holes" in your time-series, which is also bad. Pros/cons. Another reason `arbitrage`
    # is a hyper, maybe it's not worth the dirty matching.
    first = tables_[0]
    for i, table in enumerate(tables_):
        name, ts = table['name'], table['ts']
        if i == 0:
            query += f" from {name}"
            continue
        prior = tables_[i-1]
        query += f"""
            left join lateral (
              select {', '.join(c for c in table['cols'])}
              from {name}
              where {name}.{ts} <= {prior['name']}.{prior['ts']}
              order by {name}.{ts} desc
              limit 1 
            ) {name} on true
            """

    if just_count:
        query += " limit 1"
        return conn.execute(query).fetchone()[0]

    order_field = f"{first['name']}.{first['ts']}" if len(tables_) > 1 else first['ts']
    query += f" order by {order_field} desc limit {limit} offset {offset}"

    # order by date DESC (for limit to cut right), then reverse again (so old->new)
    df = pd.read_sql_query(query, conn).iloc[::-1]
    for t in tables_:
        for k, method in t['cols'].items():
            fill = {'value': 0} if method == Z else {'method': 'ffill' if method == F else 'bfill'}
            col_name = f"{t['name']}_{k}"
            df[col_name] = df[col_name].fillna(fill)
    df = df.astype('float64')

    if last_timestamp:
        # Save away last-timestamp (used in LIVE mode to inform how many new steps are added between polls
        query = f"select {first['ts']} from {first['name']} order by {first['ts']} desc limit 1"
        last_timestamp = conn.execute(query).fetchone()[first['ts']]
        return df, last_timestamp
    return df


db_to_dataframe = _db_to_dataframe_ohlc if 'coins' in DB else _db_to_dataframe_main


def fetch_more(conn, last_timestamp, arbitrage):
    """Function used to fetch more data in `live` mode in a polling loop.
    TODO this approach won't work if we switch the `arbitrage` method from OUTER JOIN to INNER (see comments in
    _db_to_dataframe_main()
    """
    t = tables[0]
    query = f"select count(*) as ct from {t['name']} where {t['ts']} > :last_timestamp"
    n_new = conn.execute(text(query), last_timestamp=last_timestamp).fetchone()['ct']
    if n_new == 0:
        return None, 0, last_timestamp
    new_data, latest_timestamp = db_to_dataframe(conn, limit=n_new, arbitrage=arbitrage, last_timestamp=True)
    return new_data, n_new, latest_timestamp


def setup_runs_table():
    """Run this function once during project setup (see README). Or just copy/paste the SQL into your runs database
    """
    conn_runs = engine_runs.connect()
    conn_runs.execute("""
        create table if not exists runs
        (
            id serial not null,
            hypers jsonb not null,
            custom_scores double precision[],
            sharpes double precision[],
            returns double precision[],
            signals double precision[],
            prices double precision[],
            uniques double precision[],
            flag varchar(16),
            agent varchar(64) default 'ppo_agent'::character varying not null
        );
    """)