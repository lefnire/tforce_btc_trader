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
        self.start_cap = 1500
        self.window = 150
        self.acc = Box(
            episode=dict(
                i=0,
                advantages=[],
                uniques=[]
            ),
            step=dict(i=0),  # set in reset()
            batch=dict(i=0)  # set in [train|test]_start()
        )
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

    def _diff(self, arr, percent=False):
        series = pd.Series(arr)
        diff = series.pct_change() if percent else series.diff()
        diff.iloc[0] = 0  # always NaN, nothing to compare to

        # Remove outliers (turn them to NaN)
        q = diff.quantile(0.99)
        diff = diff.mask(diff > q, np.nan)

        return diff.replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    def _xform_data(self, df):
        columns = []
        tables_ = data.get_tables(self.hypers.arbitrage)
        percent = self.hypers.pct_change
        for table in tables_:
            name, cols, ohlcv = table['name'], table['cols'], table.get('ohlcv', {})
            columns += [self._diff(df[f'{name}_{k}'], percent) for k in cols]

            # Add extra indicator columns
            if ohlcv and self.hypers.indicators:
                ind = pd.DataFrame()
                # TA-Lib requires specifically-named columns (OHLCV)
                for k, v in ohlcv.items():
                    ind[k] = df[f"{name}_{v}"]
                columns += [
                    ## Original indicators from boilerplate
                    self._diff(SMA(ind, timeperiod=15), percent),
                    self._diff(SMA(ind, timeperiod=60), percent),
                    self._diff(RSI(ind, timeperiod=14), percent),
                    self._diff(ATR(ind, timeperiod=14), percent),

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
        step_acc = self.acc.step
        self.cash = self.value = self.start_cap
        start_timestep = self.window if self.conv2d else 1  # advance some steps just for cushion, various operations compare back a couple steps
        step_acc.i = start_timestep
        step_acc.signals = [0] * start_timestep
        step_acc.repeats = 1
        self.acc.episode.i += 1

        # Fetch random slice of rows from the database (based on limit)
        row_ct = data.count_rows(self.conn, arbitrage=self.hypers.arbitrage)
        if self.hypers.steps == -1:
            print('Using all data')
            offset, limit = 0, 'ALL'
        else:
            offset, limit = random.randint(0, row_ct - self.hypers.steps), self.hypers.steps
        df = data.db_to_dataframe(self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage)
        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._diff(self.prices, percent=True)

        if self.conv2d:
            window = self.observations[start_timestep - self.window:start_timestep]
            first_state = dict(
                state0=self._reshape_window_for_conv2d(window),
                state1=np.array([1., 1.])
            )
        else:
            first_state = np.append(self.observations[start_timestep], [1., 1.])
        if self.hypers.scale:
            first_state = self.scaler.transform_state(first_state)
        return first_state

    def _reset_batch(self):
        self.time = time.time()
        batch_acc = self.acc.batch
        total = self.cash + self.value
        batch_acc.score = Box(start=total, end=None)
        batch_acc.hold = Box(start=total, value=self.value, cash=self.cash)

    def train_start(self):
        self.testing = False
        self._reset_batch()

    def test_start(self):
        self.testing = True
        self.acc.batch.i += 1
        self._reset_batch()

    def test_stop(self, n_test):
        time_ = round(time.time() - self.time)
        step_acc, batch_acc = self.acc.step, self.acc.batch
        batch_acc.prices = self.prices[-n_test:].tolist()
        batch_acc.signals = step_acc.signals[-n_test:]
        advantage = (batch_acc.score.end - batch_acc.score.start) - \
                    ((batch_acc.hold.value + batch_acc.hold.cash) - batch_acc.hold.start)
        self.acc.episode.advantages.append(advantage)
        self.acc.episode.uniques.append(float(len(np.unique(batch_acc.signals))))

        # Print
        common = dict((round(k), v) for k, v in Counter(batch_acc.signals).most_common(5))
        high, low = max(batch_acc.signals), min(batch_acc.signals)
        print(f"{batch_acc.i}|⌛:{time_}s\tA:{'%.3f'%advantage}\t{common}(high:{'%.2f'%high},low:{'%.2f'%low})")

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

        step_acc, batch_acc = self.acc.step, self.acc.batch

        step_acc.signals.append(float(signal))

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

        # punish every step we're in "overdraft" (the floor is lava)
        # FIXME terminate episode instead; need big changes for that
        if self.cash < 0: reward -= 500
        if self.value < 0: reward -= 500

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        diff_loc = step_acc.i if self.conv2d else step_acc.i + 1
        pct_change = self.prices_diff[diff_loc]
        self.value += pct_change * self.value
        total = self.value + self.cash
        batch_acc.score.end = total
        reward += total - before.total

        # calculate what the reward would be "if I held", to calculate the actual reward's _advantage_ over holding
        before = batch_acc.hold
        before.value += pct_change * before.value

        # Encourage diverse behavior. hypers.punish_repeats method means punishing homogenous behavior
        if self.hypers.punish_repeats:
            recent_actions = np.array(step_acc.signals[-step_acc.repeats:])
            if np.any(recent_actions > 0) and np.any(recent_actions < 0) and np.any(recent_actions == 0):
                step_acc.repeats = 1  # reset penalty-growth
            else:
                # We want a trade every x minutes. Increase punishment over time until it's 1 at that step
                # reward -= self.scaler.avg_reward() * (step_acc.repeats/(60*10))
                n_minutes = 2
                reward -= step_acc.repeats / (60 * n_minutes)
                step_acc.repeats += 1  # grow the penalty with time

        step_acc.i += 1
        # 0 if before['cash'] == 0 else (self.cash - before['cash']) / before['cash'],
        # 0 if before['value'] == 0 else (self.value - before['value']) / before['value'],
        cash_scaled, val_scaled = self.cash / self.start_cap,  self.value / self.start_cap
        if self.conv2d:
            window = self.observations[step_acc.i - self.window:step_acc.i]
            next_state = dict(
                state0=self._reshape_window_for_conv2d(window),
                state1=np.array([cash_scaled, val_scaled])
            )
        else:
            next_state = np.append(self.observations[step_acc.i], [cash_scaled, val_scaled])

        if self.hypers.scale:
            next_state = self.scaler.transform_state(next_state)
            reward = self.scaler.transform_reward(reward)

        terminal = int(step_acc.i + 1 >= len(self.observations))
        if terminal:
            step_acc.signals.append(0)  # Add one last signal (to match length)

        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def train_and_test(self, agent):
        n_rows = data.count_rows(self.conn, self.hypers.arbitrage)

        split = .85
        episodes, n_test_points = 1, 60
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
                    agent.observe(terminal=terminal, reward=reward)
                    if terminal: break
                if terminal: break

                self.test_start()
                for _ in range(n_test):
                    next_state, terminal, reward = self.execute(agent.act(next_state, deterministic=True))
                    agent.observe(terminal=terminal, reward=reward)
                    if terminal: break
                self.test_stop(n_test)

