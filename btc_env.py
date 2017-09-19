import random, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talib.abstract import SMA, RSI, ATR, EMA
from sklearn import preprocessing
from sklearn.externals import joblib
from collections import Counter
from sqlalchemy.sql import text
import tensorflow as tf
from tensorforce import util, TensorForceError
from tensorforce.environments import Environment

import data
from data import conn

try:
    scaler = joblib.load('data_/scaler.pkl')
except Exception: pass

class BitcoinEnv(Environment):
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    START_CAP = 1000

    def __init__(self, limit=10000, agent_type='DQNAgent', agent_name=None, scale_features=False, abs_reward=False,
                 indicators=False):
        """Initializes a minimal test environment."""
        self.limit = limit
        self.agent_type = agent_type
        self.agent_name = agent_name
        self.scale_features = scale_features
        self.abs_reward = abs_reward
        self.indicators = indicators

        self.continuous_actions = bool(agent_type in ['PPOAgent', 'TRPOAgent', 'NAFAgent', 'VPGAgent'])
        self.episode_results = {'cash': [], 'values': [], 'rewards': []}

    @property
    def states(self):
        return dict(shape=(self.num_features(),), type='float')

    @property
    def actions(self):
        if self.continuous_actions:
            return dict(continuous=True, shape=(), min_value=-100, max_value=100)
        else:
            return dict(continuous=False, num_actions=3)  # BUY SELL HOLD

    def num_features(self):
        num = 8 if self.indicators else 5  # num features from self._xform_data
        num *= len(data.tables)  # That many features per table
        num += 2  # [self.cash, self.value]
        return num

    def __str__(self): return 'BitcoinEnv'

    def close(self): pass

    def render(self): pass

    def seed(self, num):
        return
        # TODO is all this correct/necessary?
        random.seed(num)
        np.random.seed(num)
        tf.set_random_seed(num)

    def scale_features_and_save(self):
        """
        When using scaling on features (TODO experiment) then call this method once after changing
        any way we're using features before training
        """
        all_data, _ = self._xform_data(data.db_to_dataframe())
        # Add a rough estimate min/max of cash & value
        for i in range(2):
            all_data = np.hstack((
                all_data,
                np.random.uniform(-500000, 1500, (all_data.shape[0], 1))
            ))
        scaler = preprocessing.StandardScaler()
        scaler.fit(all_data)
        joblib.dump(scaler, 'data_/scaler.pkl')

    @staticmethod
    def pct_change(arr):
        return pd.Series(arr).pct_change()\
            .replace([np.inf, -np.inf, np.nan], [1., -1., 0.]).values

    @staticmethod
    def diff(arr):
        return pd.DataFrame(arr).diff()\
            .replace([np.inf, -np.inf], np.nan).ffill()\
            .fillna(0).values

    def _xform_data(self, df):
        if data.DB == 'coins2':
            columns = []
            for k in data.tables:
                xchange_df = df.rename(columns={
                    k + '_open': 'open',
                    k + '_high': 'high',
                    k + '_low': 'low',
                    k + '_close': 'close',
                    k + '_volume': 'volume'
                })
                columns += [
                    self.diff(xchange_df['open']),
                    self.diff(xchange_df['high']),
                    self.diff(xchange_df['low']),
                    self.diff(xchange_df['close']),
                    self.diff(xchange_df['volume']),
                ]
            states = np.nan_to_num(np.column_stack(columns))
            prices = df['g_close'].values
            return states, prices

        ### -----------

        columns = []
        for k in data.tables:
            # TA-Lib requires specifically-named columns (#TODO need to get our hands on "open")
            xchange_df = df.rename(columns={
                k + '_last': 'close',
                k + '_high': 'high',
                k + '_low': 'low',
                k + '_volume': 'volume'
            })

            # Currently NO indicators works better (LSTM learns the indicators itself). I'm thinking because indicators
            # are absolute values, causing number-range instability
            columns += [
                self.diff(xchange_df['close']),
                self.diff(xchange_df['high']),
                self.diff(xchange_df['low']),
                self.diff(xchange_df['volume']),
            ]
            if self.indicators:
                columns += [
                    ## Original indicators from boilerplate
                    # SMA(xchange_df, timeperiod=15),
                    # SMA(xchange_df, timeperiod=60),
                    # RSI(xchange_df, timeperiod=14),
                    # ATR(xchange_df, timeperiod=14),

                    ## Indicators from "How to Day Trade For a Living" (try these)
                    ## Price, Volume, 9-EMA, 20-EMA, 50-SMA, 200-SMA, VWAP, prior-day-close
                    self.diff(EMA(xchange_df, timeperiod=9)),
                    self.diff(EMA(xchange_df, timeperiod=20)),
                    self.diff(SMA(xchange_df, timeperiod=50)),
                    self.diff(SMA(xchange_df, timeperiod=200)),
                ]

        states = np.nan_to_num(np.column_stack(columns))
        prices = df['gdax_btcusd_last'].values
        # Note: don't scale/normalize here, since we'll normalize w/ self.price/self.cash after each action
        return states, prices

    def reset(self):
        self.time = time.time()
        self.cash = self.START_CAP
        self.value = self.START_CAP
        start_timestep = 1  # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        self.total_reward = 0

        # Fetch random slice of rows from the database (based on limit)
        offset = random.randint(0, data.count_rows() - self.limit)
        df = data.db_to_dataframe(limit=self.limit, offset=offset)
        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self.pct_change(self.prices)

        first_state = np.append(self.observations[start_timestep], [0., 0.])
        if self.scale_features:
            first_state = scaler.transform([first_state])[0]
        return first_state

    def execute(self, action):
        if self.continuous_actions:
            # signal = 0 if -40 < action < 5 else action
            signal = 0 if -1 < action < 1 else action
        else:
            signal = 40 if action == self.ACTION_BUY\
                else -40 if action == self.ACTION_SELL\
                else 0

        self.signals.append(signal)

        abs_sig = abs(signal)
        fee = 0.0025  # https://www.gdax.com/fees/BTC-USD
        before = dict(cash=self.cash, value=self.value, total=self.cash+self.value)
        if signal > 0:
            if self.cash >= abs_sig:
                self.value += abs_sig - abs_sig*fee
            self.cash -= abs_sig
        elif signal < 0:
            if self.value >= abs_sig:
                self.cash += abs_sig - abs_sig*fee
            self.value -= abs_sig

        pct_change = self.prices_diff[self.timestep + 1]  # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        self.value += pct_change * self.value
        total = self.value + self.cash
        if self.abs_reward:
            reward = total - self.START_CAP*2  # Absolute reward
        else:
            reward = total - before['total']  # Relative reward (seems to work better)

        self.timestep += 1
        next_state = np.append(self.observations[self.timestep], [
            self.cash,  # 0 if before['cash'] == 0 else (self.cash - before['cash']) / before['cash'],
            self.value  # 0 if before['value'] == 0 else (self.value - before['value']) / before['value'],
        ])
        if self.scale_features:
            next_state = scaler.transform([next_state])[0]

        self.total_reward += reward

        terminal = int(self.timestep + 1 >= len(self.observations))
        if terminal:
            self.time = round(time.time() - self.time)
            self.signals.append(0)  # Add one last signal (to match length)
            self.episode_results['cash'].append(self.cash)
            self.episode_results['values'].append(self.value)
            self.episode_results['rewards'].append(self.total_reward)
            self.action_counter = dict((round(k), v) for k, v in Counter(self.signals).most_common(5))
            self.write_results()
        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, reward, terminal

    step = execute  # alias execute as step, called step by some frameworks

    def write_results(self):
        episode = len(self.episode_results['cash'])
        reward, cash, value = self.total_reward, self.cash, self.value
        print("{}) time:{}, reward:{} totals:{}, actions:{}".format(
            episode, self.time, round(reward), round(cash + value), self.action_counter))

        # save a snapshot of the actual graph & the buy/sell signals so we can visualize elsewhere
        if cash + value > BitcoinEnv.START_CAP * 2:
            y = list(self.prices)
            signals = list(self.signals)
        else:
            y = None
            signals = None

        q = text("""
                insert into episodes (episode, reward, cash, value, agent_name, steps, y, signals) 
                values (:episode, :reward, :cash, :value, :agent_name, :steps, :y, :signals)
                -- Don't overwrite, in case we're using A3C with 7 workers doing the same work - just record one
                -- worker's efforts. Because of this, make sure to drop the table elsewhere before 
                on conflict do nothing;
            """)
        conn.execute(q, episode=episode, reward=reward, cash=cash, value=value, agent_name=self.agent_name,
                     steps=self.timestep, y=y, signals=signals)