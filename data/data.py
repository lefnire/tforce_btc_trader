import time, json, re, pdb
from os import path
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text
from utils import raise_refactor, last_good_commit
from sklearn.preprocessing import robust_scale
import os

# From connecting source file, `import engine` and run `engine.connect()`. Need each connection to be separate
# (see https://stackoverflow.com/questions/3724900/python-ssl-problem-with-multiprocessing)
config_json = json.load(open(os.path.dirname(__file__) + '/../config.json'))
DB = config_json['DB_HISTORY'].split('/')[-1]
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

# see {last_good_commit} for imputes (ffill, bfill, zero),
# alex database

def setup_runs_table():
    """Run this function once during project setup (see README). Or just copy/paste the SQL into your runs database
    """
    conn_runs = engine_runs.connect()
    conn_runs.execute("""
        create table if not exists runs
        (
            id uuid not null,
            hypers jsonb not null,
            returns double precision[],
            signals double precision[],
            prices double precision[],
            uniques double precision[]
        );
    """)

class Data(object):
    def __init__(self, ep_len=5000, window=300, arbitrage=False, indicators={}):
        self.ep_len = ep_len
        self.window = window
        self.arbitrage = arbitrage
        self.indicators = indicators

        self.ep_stride = ep_len  # disjoint
        # self.ep_stride = 100  # overlap; shift each episode by x seconds.
        # TODO overlapping stride would cause test/train overlap. Tweak it so train can overlap data, but test gets silo'd

        col_renames = {
            'Timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume_(BTC)': 'volume_btc',
            'Volume_(Currency)': 'volume',
            'Weighted_Price': 'vwap'
        }

        filenames = {
            # 'bitstamp': 'bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv',
            'coinbase': 'coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv',
            # 'coincheck': 'coincheckJPY_1-min_data_2014-10-31_to_2018-06-27.csv'
        }
        primary_table = 'coinbase'
        self.target = f"{primary_table}_close"

        df = None
        for table, filename in filenames.items():
            df_ = pd.read_csv(path.join(path.dirname(__file__), 'bitcoin-historical-data', filename))
            col_renames_ = {k: f"{table}_{v}" for k, v in col_renames.items()}
            df_ = df_.rename(columns=col_renames_)
            ts = f"{table}_timestamp"
            df_[ts] = pd.to_datetime(df_[ts], unit='s')
            df_ = df_.set_index(ts)
            df = df_ if df is None else df.join(df_)

        # too quiet before 2015, time waste. copy() to avoid pandas errors
        df = df.loc['2015':].copy()

        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour

        # TODO drop null rows? (inner join?)
        # TODO arbitrage
        # TODO indicators

        diff_cols = [
            f"{table}_{k}" for k in
            'open high low close volume_btc volume vwap'.split(' ')
            for table in filenames.keys()
        ]
        df[diff_cols] = df[diff_cols].pct_change()\
            .replace([np.inf, -np.inf], np.nan)\
            .ffill()  # .bfill()?
        df = df.iloc[1:]
        target = df[self.target]  # don't scale price changes; we use that in raw form later
        df = pd.DataFrame(
            robust_scale(df.values, quantile_range=(.1, 100-.1)),
            columns=df.columns, index=df.index
        )
        df[self.target] = target

        df['cash'], df['value'] = 0., 0.

        self.df = df

    def offset(self, ep, step):
        return ep * self.ep_stride + step

    def has_more(self, ep):
        return self.offset(ep + 2, 0) + self.window < self.df.shape[0]
        # return (ep + 1) * self.ep_stride + self.window < self.df.shape[0]

    def get_data(self, ep, step):
        offset = self.offset(ep, step)
        X = self.df.iloc[offset:offset+self.window]
        y = self.df.iloc[offset+self.window]
        return X, y

    def get_prices(self, ep, step):
        offset = self.offset(ep, step)
        return self.df.iloc[offset + self.window:self.ep_len][self.target]

    def reset_cash_val(self):
        self.df['cash'] = 0.
        self.df['value'] = 0.

    def set_cash_val(self, ep, step, cash, value):
        offset = self.offset(ep, step)
        self.df.cash.iloc[offset] = cash
        self.df.value.iloc[offset] = value

    def fetch_more(self):
        raise_refactor()
