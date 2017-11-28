import random, time, gym, re, json
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talib.abstract import SMA, RSI, ATR, EMA
from sklearn import preprocessing
from sklearn.externals import joblib
from collections import Counter
from sqlalchemy.sql import text
import tensorflow as tf
from box import Box
from tensorforce import util, TensorForceError
from tensorforce.environments import Environment
import pdb

ALLOW_SEED = False

import data
from data import engine, NCOL

try:
    scaler = joblib.load('saves/scaler.pkl')
except Exception: pass

try:
    min_max = joblib.load('saves/min_max.pkl')
    min_max_scaled = joblib.load('saves/min_max_scaled.pkl')
except Exception:
    min_max = None


class BitcoinEnv(Environment):
    def __init__(self, hypers, name='ppo_agent', write_graph=False, log_states=False):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.hypers = Box(hypers)
        self.conv2d = self.hypers.net_type == 'conv2d'
        self.agent_name = name
        self.start_cap = 1e3
        self.window = 150
        self.write_graph = write_graph
        self.log_states = log_states
        self.episode_results = {'cash': [], 'values': [], 'rewards': []}

        self.conn = engine.connect()

        # Action space
        if re.search('(dqn|ppo|a3c)', name, re.IGNORECASE):
            self.actions_ = dict(type='int', shape=(), num_actions=5)
        else:
            self.actions_ = dict(type='float', shape=(), min_value=-100, max_value=100)

        # Observation space
        if self.conv2d:
            # width = window width (150 time-steps)
            # height = num_features, but one layer for each table
            # channels is number of tables (each table is a "layer" of features
            self.states_ = dict(
                state0=dict(type='float', min_value=-1, max_value=1, shape=(150,NCOL,2)),  # image 150x7x2-dim
                state1=dict(type='float', min_value=-1, max_value=1, shape=(2))  # money
            )
        else:
            if self.hypers.scale:
                self.states_ = dict(type='float', min_value=min_max_scaled[0], max_value=min_max_scaled[1], shape=())
                print('using min_max', min_max_scaled)
            elif min_max:
                self.states_ = dict(type='float', min_value=min_max[0], max_value=min_max[1], shape=())
                print('using min_max', min_max)
            else:
                default_min_max = 1 if self.hypers.diff_percent else 1
                self.states_ = dict(type='float', min_value=-default_min_max, max_value=default_min_max, shape=(self.num_features(),))

        # self._seed()
        if write_graph:
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
            self.summary_writer = tf.summary.FileWriter(f"saves/boards/{self.agent_name}")
            self.signals_placeholder = tf.placeholder(tf.float16, shape=(None,))
            tf.summary.histogram('buy_sell_signals', self.signals_placeholder, collections=['btc_env'])
            self.merged_summaries = tf.summary.merge_all('btc_env')
            data.wipe_rows(self.conn, name)

    def __str__(self): return 'BitcoinEnv'
    def close(self): self.conn.close()
    @property
    def states(self): return self.states_
    @property
    def actions(self): return self.actions_
    def seed(self, seed=None):
        if not ALLOW_SEED: return
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def num_features(self):
        num = len(data.columns)
        if self.hypers.indicators:
            num += 4  # num features from self._get_indicators
        num *= len(data.tables)  # That many features per table
        num += 2  # [self.cash, self.value]
        return num

    def _pct_change(self, arr):
        return pd.Series(arr).pct_change()\
            .replace([np.inf, -np.inf, np.nan], [1., -1., 0.]).values

    def _diff(self, arr):
        if self.hypers.diff_percent:
            return self._pct_change(arr)
        return pd.Series(arr).diff()\
            .replace([np.inf, -np.inf], np.nan).ffill()\
            .fillna(0).values

    def _xform_data(self, df):
        columns = []
        for k in data.tables:
            # TA-Lib requires specifically-named columns (OHLCV)
            c = dict([(f'{k}_{c}', c) for c in data.columns if c != data.close_col])
            c[f'{k}_{data.close_col}'] = 'close'
            xchange_df = df.rename(columns=c)

            # Currently NO indicators works better (LSTM learns the indicators itself). I'm thinking because indicators
            # are absolute values, causing number-range instability
            columns += list(map(lambda k: self._diff(xchange_df[k]), c.values()))
            if self.hypers.indicators:
                columns += self._get_indicators(xchange_df)

        states = np.nan_to_num(np.column_stack(columns))
        prices = df[data.predict_col].values
        # Note: don't scale/normalize here, since we'll normalize w/ self.price/self.cash after each action
        return states, prices

    def _get_indicators(self, df):
        return [
            ## Original indicators from boilerplate
            self._diff(SMA(df, timeperiod=15)),
            self._diff(SMA(df, timeperiod=60)),
            self._diff(RSI(df, timeperiod=14)),
            self._diff(ATR(df, timeperiod=14)),

            ## Indicators from "How to Day Trade For a Living" (try these)
            ## Price, Volume, 9-EMA, 20-EMA, 50-SMA, 200-SMA, VWAP, prior-day-close
            # self._diff(EMA(df, timeperiod=9)),
            # self._diff(EMA(df, timeperiod=20)),
            # self._diff(SMA(df, timeperiod=50)),
            # self._diff(SMA(df, timeperiod=200)),
        ]

    def reset(self):
        self.time = time.time()
        self.cash = self.value = self.start_cap
        start_timestep = self.window if self.conv2d else 1  # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        self.total_reward = 0

        # Fetch random slice of rows from the database (based on limit)
        offset = random.randint(0, data.count_rows(self.conn) - self.hypers.steps)
        df = data.db_to_dataframe(self.conn, limit=self.hypers.steps, offset=offset)
        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._pct_change(self.prices)

        if self.conv2d:
            window = self.observations[self.timestep - self.window:self.timestep]
            # TODO adapt to more than 2 tables
            first_state = dict(
                state0=np.transpose([window[:, 0:NCOL], window[:, NCOL:]], (1,2,0)),
                state1=np.array([1., 1.])
            )
        else:
            first_state = np.append(self.observations[start_timestep], [1., 1.])
            if self.hypers.scale:
                first_state = scaler.transform([first_state])[0]
        return first_state

    def execute(self, actions):
        if self.actions['type'] == 'int':
            signals = {
                0: -40,
                1: 0,
                2: 1,
                3: 5,
                4: 20
            }
            signal = signals[actions]
        else:
            signal = actions
            # gdax requires a minimum .01BTC (~$40) sale. As we're learning a probability distribution, don't separate
            # it w/ cutting -40-0 as 0 - keep it "continuous" by augmenting like this.
            if signal < 0: signal -= 40
        self.signals.append(signal)

        fee = 0.0025  # https://www.gdax.com/fees/BTC-USD
        abs_sig = abs(signal)
        before = self.cash + self.value
        if signal > 0:
            if self.cash >= abs_sig:
                self.value += abs_sig - abs_sig*fee
            self.cash -= abs_sig
        elif signal < 0:
            if self.value >= abs_sig:
                self.cash += abs_sig - abs_sig*fee
            self.value -= abs_sig

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        diff_loc = self.timestep if self.conv2d else self.timestep + 1
        pct_change = self.prices_diff[diff_loc]
        self.value += pct_change * self.value
        total = self.value + self.cash
        reward = total - before  # Relative reward (seems to work better)

        self.timestep += 1
        # 0 if before['cash'] == 0 else (self.cash - before['cash']) / before['cash'],
        # 0 if before['value'] == 0 else (self.value - before['value']) / before['value'],
        cash_scaled, val_scaled = self.cash / self.start_cap,  self.value / self.start_cap
        if self.conv2d:
            window = self.observations[self.timestep - self.window:self.timestep]
            next_state = dict(
                state0=np.transpose([window[:, 0:NCOL], window[:, NCOL:]], (1,2,0)),
                state1=np.array([cash_scaled, val_scaled])
            )

        else:
            next_state = np.append(self.observations[self.timestep], [cash_scaled, val_scaled])

        # If we need to record a few thousand observations for use in scaling or determining min/max vals (turn off after)
        if self.log_states:
            # create table if not exists observations (obs double precision[])
            obs = [float(o) for o in next_state]
            self.conn.execute(text("insert into observations (obs) values (:obs)"), obs=obs)

        if self.hypers.scale:
            next_state = scaler.transform([next_state])[0]

        # Punish in-action options: hold_* vs unique_* means "he's only holding" or "he's doing the same thing 
        # over and over". *_double means "double the reward (up the ante)" or "*_spank" means "just punish him"
        punish, recent_actions = self.hypers.punish_inaction, self.signals[-100:]
        if ('unique' in punish and np.unique(recent_actions).size == 1)\
            or ('hold' in punish and (np.array(recent_actions) == 0).all()):
            if 'double' in punish:
                reward *= 2  # up the ante
            elif 'spank' in punish:
                reward -= 5  # just penalize

        self.total_reward += reward

        terminal = int(self.timestep + 1 >= len(self.observations))
        if terminal:
            self.signals.append(0)  # Add one last signal (to match length)
            self.episode_results['cash'].append(self.cash)
            self.episode_results['values'].append(self.value)
            self.episode_results['rewards'].append(self.total_reward)
            self.time = round(time.time() - self.time)
            self._write_results()
        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def _write_results(self):
        res = self.episode_results
        episode = len(res['cash'])
        if episode % 5 != 0: return  # TODO temporary: reduce the output clutter
        reward, cash, value = float(self.total_reward), float(self.cash), float(self.value)
        avg50 = round(np.mean(res['rewards'][-50:]))
        common = dict((round(k), v) for k, v in Counter(self.signals).most_common(5))
        high, low = np.max(self.signals), np.min(self.signals)
        print(f"{episode}\t⌛:{self.time}s\tR:{int(reward)}\tavg50:{avg50}\tA:{common}(high={high},low={low})")

        if self.write_graph:
            scalar = tf.Summary()
            scalar.value.add(tag='perf/time', simple_value=self.time)
            scalar.value.add(tag='perf/reward_eps', simple_value=float(self.total_reward))
            scalar.value.add(tag='perf/reward', simple_value=reward)
            scalar.value.add(tag='perf/reward50avg', simple_value=avg50)
            self.summary_writer.add_summary(scalar, episode)

            # Every so often, record signals distribution
            if episode % 10 == 0:
                histos = self.sess.run(self.merged_summaries, feed_dict={self.signals_placeholder: self.signals})
                self.summary_writer.add_summary(histos, episode)

            self.summary_writer.flush()


def scale_features_and_save():
    """
    If we want to scale/normalize or min/max features (states), first run the Env in log_states=True mode for a while,
    then call this function manually from python shell
    """
    conn = engine.connect()
    observations = conn.execute('select obs from observations').fetchall()
    observations = [o[0] for o in observations]
    conn.close()

    mat = np.array(observations)
    min_max = [np.floor(np.amin(mat, axis=0)), np.ceil(np.amax(mat, axis=0))]
    print('min/max: ', min_max)
    joblib.dump(min_max, 'saves/min_max.pkl')

    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(observations)
    joblib.dump(scaler, 'saves/scaler.pkl')

    mat = np.array(scaled)
    min_max = [np.floor(np.amin(mat, axis=0)), np.ceil(np.amax(mat, axis=0))]
    print('min/max (scaled): ', min_max)
    joblib.dump(min_max, 'saves/min_max_scaled.pkl')
