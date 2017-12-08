import random, time, re, math
import numpy as np
import pandas as pd
from talib.abstract import SMA, RSI, ATR, EMA
from collections import Counter
from sqlalchemy.sql import text
import tensorflow as tf
from box import Box
from tensorforce.environments import Environment
from sklearn.preprocessing import RobustScaler, robust_scale
import pdb

ALLOW_SEED = False

import data
from data import engine


class Scaler(object):
    STOP_AT = 3e5  # 400k is size of table. Might be able to do with much less, being on safe side
    SKIP = 15
    def __init__(self):
        self.reward_scaler = RobustScaler()
        self.state_scaler = RobustScaler()
        self.rewards = []
        self.states = []
        self.done = False
        self.i = 0

    def _should_skip(self):
        # After we've fitted enough (see STOP_AT), start returning direct-transforms for performance improvement
        # Skip every few fittings. Each individual doesn't contribute a whole lot anyway, and costs a lot
        return self.done or (self.i % self.SKIP != 0 and self.i > self.SKIP)

    def transform_state(self, state):
        self.i += 1
        if self._should_skip():
            return self.state_scaler.transform([state])[-1]
        # Fit, transform, return
        self.states.append(state)
        ret = self.state_scaler.fit_transform(self.states)[-1]
        if self.i >= self.STOP_AT:
            # Clear up memory, fitted scalers have all the info we need. stop=True only needed in one of these functions
            del self.rewards
            del self.states
            self.done = True
        return ret

    def transform_reward(self, reward):
        if self._should_skip():
            return self.reward_scaler.transform([[reward]])[-1][0]
        self.rewards.append([reward])
        return self.reward_scaler.fit_transform(self.rewards)[-1][0]

# keep this globally arround for all runs forever
scaler = Scaler()
scaler_indicators = Scaler()


class BitcoinEnv(Environment):
    def __init__(self, hypers, name='ppo_agent'):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.hypers = Box(hypers)
        self.conv2d = self.hypers['net.type'] == 'conv2d'
        self.agent_name = name
        self.start_cap = 1e3
        self.window = 150
        self.episode_rewards = []
        self.scaler = scaler_indicators if self.hypers.indicators else scaler

        self.conn = engine.connect()

        # Action space
        if self.hypers.unimodal:
            self.actions_ = dict(type='float', shape=(), min_value=-100, max_value=100)
        else:
            self.actions_ = dict(
                action=dict(type='int', shape=(), num_actions=3),
                amount=dict(type='float', shape=(), min_value=-100, max_value=100))

        # Observation space
        self.cols_ = self.n_cols(conv2d=self.conv2d, indicators=self.hypers.indicators)
        if self.conv2d:
            # width = window width (150 time-steps)
            # height = num_features, but one layer for each table
            # channels is number of tables (each table is a "layer" of features
            self.states_ = dict(
                state0=dict(type='float', min_value=-1, max_value=1, shape=(150, self.cols_, len(data.tables))),  # image 150xNCOLx2
                state1=dict(type='float', min_value=-1, max_value=1, shape=2)  # money
            )
        else:
            self.states_ = dict(type='float', shape=self.cols_)

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

    @staticmethod
    def n_cols(conv2d=True, indicators=False):
        num = data.LEN_COLS
        if indicators:
            num += data.N_INDICATORS  # num features from self._get_indicators
        if not conv2d: # These will be added in downstream Dense
            # num *= len(data.tables)  # That many features per table
            num += 2  # [self.cash, self.value]
        return num

    def _pct_change(self, arr):
        change = pd.Series(arr).pct_change()
        change.iloc[0] = 0  # always NaN, nothing to compare to
        if change.isin([np.nan, np.inf, -np.inf]).any():
            raise Exception(f"NaN or inf detected in _pct_change()!")
        return change.values

    def _diff(self, arr, fill=False):
        diff = pd.Series(arr).diff()
        diff.iloc[0] = 0  # always NaN, nothing to compare to
        if diff.isin([np.nan, np.inf, -np.inf]).any():
            if fill:  # if caller is explicitly OK with 0-fills (eg, SMA)
                diff.fillna(0, inplace=True)
            else:
                raise Exception(f"NaN or inf detected in _diff()!")
        return diff.values
        # return diff.ffill(inplace=True).fillna(0).values

    def _xform_data(self, df):
        columns = []
        for table in data.tables:
            name, cols, ohlcv = table['name'], table['columns'], table.get('ohlcv', {})
            # TA-Lib requires specifically-named columns (OHLCV)
            c = dict([(f'{name}_{c}', c) for c in cols if c not in ohlcv.values()])
            for k, v in ohlcv.items():
                c[f'{name}_{v}'] = k
            xchange_df = df.rename(columns=c)

            # Currently NO indicators works better (LSTM learns the indicators itself). I'm thinking because indicators
            # are absolute values, causing number-range instability
            columns += list(map(lambda k: self._diff(xchange_df[k]), c.values()))
            if self.hypers.indicators and ohlcv:
                columns += self._get_indicators(xchange_df)

        states = np.nan_to_num(np.column_stack(columns))
        prices = df[data.target].values
        # Note: don't scale/normalize here, since we'll normalize w/ self.price/self.cash after each action
        return states, prices

    def _get_indicators(self, df):
        return [
            ## Original indicators from boilerplate
            self._diff(SMA(df, timeperiod=15), fill=True),
            self._diff(SMA(df, timeperiod=60), fill=True),
            self._diff(RSI(df, timeperiod=14), fill=True),
            self._diff(ATR(df, timeperiod=14), fill=True),

            ## Indicators from "How to Day Trade For a Living" (try these)
            ## Price, Volume, 9-EMA, 20-EMA, 50-SMA, 200-SMA, VWAP, prior-day-close
            # self._diff(EMA(df, timeperiod=9)),
            # self._diff(EMA(df, timeperiod=20)),
            # self._diff(SMA(df, timeperiod=50)),
            # self._diff(SMA(df, timeperiod=200)),
        ]

    def _reshape_window_for_conv2d(self, window):
        if len(data.tables) == 1:
            return np.expand_dims(window, -1)
        elif len(data.tables) == 2:  # default (risk arbitrage)
            return np.transpose([window[:, 0:self.cols_], window[:, self.cols_:]], (1, 2, 0))
        else:
            raise NotImplementedError('TODO Implement conv2d features depth > 2')

    def reset(self):
        self.time = time.time()
        self.cash = self.value = self.start_cap
        start_timestep = self.window if self.conv2d else 1  # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        self.total_reward = 0
        self.repeat_ct = 1

        # Fetch random slice of rows from the database (based on limit)
        offset = random.randint(0, data.count_rows(self.conn) - self.hypers.steps)
        df = data.db_to_dataframe(self.conn, limit=self.hypers.steps, offset=offset)
        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._pct_change(self.prices)

        if self.conv2d:
            window = self.observations[self.timestep - self.window:self.timestep]
            first_state = dict(
                state0=self._reshape_window_for_conv2d(window),
                state1=np.array([1., 1.])
            )
        else:
            first_state = np.append(self.observations[start_timestep], [1., 1.])
        if self.hypers.scale:
            first_state = self.scaler.transform_state(first_state)
        return first_state

    def execute(self, actions):
        min_trade = 24  # gdax min order size = .01btc ($118); krakken = .002btc ($23.60)
        if self.hypers.unimodal:
            signal = 0 if -min_trade < actions < min_trade else actions
        else:
            signal = {
                0: -1,  # make amount negative
                1: 0,  # hold
                2: 1  # make amount positive
            }[actions['action']] * actions['amount']
            if not signal: signal = 0  # sometimes gives -0.0, dunno if that matters anywhere downstream
            elif signal < 0: signal -= min_trade
            elif signal > 0: signal += min_trade

        self.signals.append(float(signal))

        fee = 0.0026  # https://www.kraken.com/en-us/help/fees
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
        reward = total - before

        # Add the "pure" reward now to the total, which is used for human analysis. We'll tweak the reward for
        # agent optimization going forward
        self.total_reward += reward

        # Encourage diverse behavior by punishing the same consecutive action. See 8741ff0 for prior ways I explored
        # this, including: (1) no interference (2) punish for holding too long (3) punish for repeats (4) instead
        # of punishing, "up the ante" by doubling the reward (kinda force him to take a closer look). Too many options
        # had dimensionality down-side, so I'm trying an "always on" but "vary the max #steps" approach.
        recent_actions = np.array(self.signals[-self.repeat_ct:])
        if np.any(recent_actions > 0) and np.any(recent_actions < 0) and np.any(recent_actions == 0):
            self.repeat_ct = 1  # reset penalty-growth
        else:
            reward -= self.repeat_ct / 50
            self.repeat_ct += 1  # grow the penalty with time

        self.timestep += 1
        # 0 if before['cash'] == 0 else (self.cash - before['cash']) / before['cash'],
        # 0 if before['value'] == 0 else (self.value - before['value']) / before['value'],
        cash_scaled, val_scaled = self.cash / self.start_cap,  self.value / self.start_cap
        if self.conv2d:
            window = self.observations[self.timestep - self.window:self.timestep]
            next_state = dict(
                state0=self._reshape_window_for_conv2d(window),
                state1=np.array([cash_scaled, val_scaled])
            )
        else:
            next_state = np.append(self.observations[self.timestep], [cash_scaled, val_scaled])

        if self.hypers.scale:
            next_state = self.scaler.transform_state(next_state)
            reward = self.scaler.transform_reward(reward)

        terminal = int(self.timestep + 1 >= len(self.observations))
        if terminal:
            self.signals.append(0)  # Add one last signal (to match length)
            self.time = round(time.time() - self.time)
            # average so we can compare runs w/ varying steps
            avg_reward = float(self.total_reward / self.hypers.steps)
            # Clamp to reasonable bounds (data corruption). How-to automatic calculation rather than hard-coded?
            # RobustScaler handles outliers for agent; this is for human & GP. float() b/c numpy=>psql
            avg_reward = float(np.clip(avg_reward, -200, 200))
            self.episode_rewards.append(avg_reward)
            self._write_results()
        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def _write_results(self):
        rewards = self.episode_rewards
        episode = len(rewards)
        if episode % 5 != 0: return
        common = dict((round(k), v) for k, v in Counter(self.signals).most_common(5))
        reward, high, low = rewards[-1], max(self.signals), min(self.signals)
        print(f"{episode}\t⌛:{self.time}s\tR:{'%.2f'%reward}\tA:{common}(high={'%.2f'%high},low={'%.2f'%low})")
