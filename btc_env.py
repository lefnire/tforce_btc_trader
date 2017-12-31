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
        self.reward_scaler = RobustScaler(quantile_range=(5., 95.))
        self.state_scaler = RobustScaler(quantile_range=(5., 95.))
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

    def avg_reward(self):
        if self.i < self.SKIP: return 20
        reward = self.reward_scaler.inverse_transform([[0]])[-1][0]
        return abs(reward)

# keep this globally around for all runs forever
scalers = {}


class BitcoinEnv(Environment):
    def __init__(self, hypers, name='ppo_agent'):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.hypers = Box(hypers)
        self.conv2d = self.hypers['net.type'] == 'conv2d'
        self.agent_name = name
        self.start_cap = 1e3
        self.window = 150
        self.episode = 0
        self.reward_avgs = dict(reward=[], advantage=[], score=[])
        self.episode_uniques = []
        self.testing = False
        self.conn = engine.connect()

        # Action space
        if self.hypers.unimodal:
            self.actions_ = dict(type='float', shape=(), min_value=-100, max_value=100)
        else:
            self.actions_ = dict(
                action=dict(type='int', shape=(), num_actions=3),
                amount=dict(type='float', shape=(), min_value=-100, max_value=100))

        # Observation space
        self.cols_ = data.n_cols(conv2d=self.conv2d, indicators=self.hypers.indicators, arbitrage=self.hypers.arbitrage)
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

        scaler_k = f'ind={self.hypers.indicators}arb={self.hypers.arbitrage}'
        if scaler_k not in scalers:
            scalers[scaler_k] = Scaler()
        self.scaler = scalers[scaler_k]

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

    def _pct_change(self, arr):
        change = pd.Series(arr).pct_change()
        change.iloc[0] = 0  # always NaN, nothing to compare to
        if change.isin([np.nan, np.inf, -np.inf]).any():
            raise Exception(f"NaN or inf detected in _pct_change()!")
        return change.values

    def _diff(self, arr):
        series = pd.Series(arr)
        diff = series.pct_change() if self.hypers.pct_change else series.diff()
        diff.iloc[0] = 0  # always NaN, nothing to compare to
        return diff.replace([np.inf, -np.inf], np.nan).ffill().fillna(0).values

    def _xform_data(self, df):
        columns = []
        tables_ = data.get_tables(self.hypers.arbitrage)
        for table in tables_:
            name, cols, ohlcv = table['name'], table['cols'], table.get('ohlcv', {})
            columns += [self._diff(df[f'{name}_{k}']) for k in cols]

            # Add extra indicator columns
            if ohlcv and self.hypers.indicators:
                ind = pd.DataFrame()
                # TA-Lib requires specifically-named columns (OHLCV)
                for k, v in ohlcv.items():
                    ind[k] = df[f"{name}_{v}"]
                columns += [
                    ## Original indicators from boilerplate
                    self._diff(SMA(ind, timeperiod=15)),
                    self._diff(SMA(ind, timeperiod=60)),
                    self._diff(RSI(ind, timeperiod=14)),
                    self._diff(ATR(ind, timeperiod=14)),

                    ## Indicators from "How to Day Trade For a Living" (try these)
                    ## Price, Volume, 9-EMA, 20-EMA, 50-SMA, 200-SMA, VWAP, prior-day-close
                    # self._diff(EMA(ind, timeperiod=9)),
                    # self._diff(EMA(ind, timeperiod=20)),
                    # self._diff(SMA(ind, timeperiod=50)),
                    # self._diff(SMA(ind, timeperiod=200)),
                ]

        states = np.nan_to_num(np.column_stack(columns))
        prices = df[data.target].values
        # Note: don't scale/normalize here, since we'll normalize w/ self.price/self.cash after each action
        return states, prices

    def _reshape_window_for_conv2d(self, window):
        if len(data.tables) == 1:
            return np.expand_dims(window, -1)
        elif len(data.tables) == 2:  # default (risk arbitrage)
            return np.transpose([window[:, 0:self.cols_], window[:, self.cols_:]], (1, 2, 0))
        else:
            raise NotImplementedError('TODO Implement conv2d features depth > 2')

    def reset(self):
        self.cash = self.value = self.start_cap
        start_timestep = self.window if self.conv2d else 1  # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        self.rewards = dict(reward=[], advantage=[], score=[])
        self.repeat_ct = 1

        # Fetch random slice of rows from the database (based on limit)
        row_ct = data.count_rows(self.conn, arbitrage=self.hypers.arbitrage)
        if self.hypers.steps == -1:
            print('Using all data')
            offset, limit = 0, 'ALL'
        else:
            offset, limit = random.randint(0, row_ct - self.hypers.steps), self.hypers.steps
        df = data.db_to_dataframe(self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage)
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

    def train_start(self):
        self.testing = False

    def test_start(self):
        self.testing = True
        self.cash = self.value = self.start_cap  # FIXME this will likely confuse the LSTM (suddenly I have $ again?) Reconsider
        self.episode += 1
        self.time = time.time()

    def test_stop(self, n_test):
        self.time = round(time.time() - self.time)
        signals = self.signals[-n_test:]
        for k in ['reward', 'advantage']:
            # - average so we can compare runs w/ varying steps
            # - Clamp to reasonable bounds (data corruption). How-to automatic calculation rather than hard-coded?
            #   RobustScaler handles outliers for agent; this is for human & GP. float() b/c numpy=>psql
            avg = np.mean(self.rewards[k][-n_test:])
            avg = float(np.clip(avg, -1000, 30))
            self.reward_avgs[k].append(avg)
        self.reward_avgs['score'].append(self.rewards['score'][-1] - self.rewards['score'][-n_test])
        self.episode_uniques.append(float(len(np.unique(signals))))

        # Print
        reward, advantage, score = self.reward_avgs['reward'][-1], \
                                   self.reward_avgs['advantage'][-1], \
                                   self.reward_avgs['score'][-1]
        common = dict((round(k), v) for k, v in Counter(signals).most_common(5))
        high, low = max(signals), min(signals)
        print(f"{self.episode}|⌛:{self.time}s\tR:{'%.2f'%reward}\tA:{'%.2f'%advantage}\tS:{'%.2f'%score}\t{common}(high={'%.2f'%high},low={'%.2f'%low})")

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
        reward = 0
        abs_sig = abs(signal)
        before = Box(cash=self.cash, value=self.value, total=self.cash+self.value)
        if signal > 0:
            self.value += abs_sig - abs_sig*fee
            self.cash -= abs_sig
        elif signal < 0:
            self.cash += abs_sig - abs_sig*fee
            self.value -= abs_sig
        if self.cash < 0 or self.value < 0:
            reward -= 1

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        diff_loc = self.timestep if self.conv2d else self.timestep + 1
        pct_change = self.prices_diff[diff_loc]
        self.value += pct_change * self.value
        total = self.value + self.cash
        reward += total - before.total
        self.rewards['reward'].append(reward)
        self.rewards['score'].append(total)

        # calculate what the reward would be "if I held", to calculate the actual reward's _advantage_ over holding
        if_i_held_value = before.value + pct_change * before.value
        if_i_held_reward = (if_i_held_value + before.cash) - before.total
        advantage = reward - if_i_held_reward  # as far as the agent is concerned
        self.rewards['advantage'].append(advantage)

        reward = reward + advantage

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
            # reward = self.scaler.transform_reward(reward)

        terminal = int(self.timestep + 1 >= len(self.observations))
        if terminal:
            self.signals.append(0)  # Add one last signal (to match length)

        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def train_and_test(self, agent):
        n_rows = data.count_rows(self.conn, self.hypers.arbitrage)

        split = .85
        episodes, n_test_points = 3, 60
        n_test_points = n_test_points / episodes
        n_train = int(n_rows * split / n_test_points)
        n_test = int(n_rows * (1-split) / n_test_points)

        for _ in range(episodes):
            agent.reset()
            next_state, terminal = self.reset(), False
            while not terminal:
                self.train_start()
                for _ in range(n_train):
                    next_state, terminal, reward = self.execute(agent.act(next_state))
                    if terminal: break
                if terminal: break

                self.test_start()
                for _ in range(n_test):
                    next_state, terminal, reward = self.execute(agent.act(next_state, deterministic=True))
                    if terminal: break
                self.test_stop(n_test)

