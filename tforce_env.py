import random, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talib.abstract import SMA, RSI, ATR
from sklearn import preprocessing
from tradingWithPython.lib.backtest import Backtest
from collections import Counter

import tensorflow as tf
from tensorforce import util, TensorForceError
from tensorforce.environments import Environment

import helpers
from helpers import config

class BitcoinEnv(Environment):
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    START_CAP = 1000
    USE_INDICATORS = False

    def __init__(self, limit=10000, agent_type='DQNAgent', agent_name=None):
        """Initializes a minimal test environment."""
        self.continuous_actions = bool(agent_type in ['PPOAgent', 'TRPOAgent', 'NAFAgent'])
        self.limit = limit
        self.agent_type = agent_type
        self.name = agent_name
        self.episode_cashs = []
        self.episode_values = []


    @staticmethod
    def num_features():
        num = 7 if BitcoinEnv.USE_INDICATORS else 4  # see xform_data for which features in either case
        num *= len(helpers.tables)  # That many features per table
        num += 2  # [self.cash, self.value]
        return num

    @staticmethod
    def pct_change(arr):
        return pd.Series(arr).pct_change()\
            .replace([np.inf, -np.inf, np.nan], [1., -1., 0.]).values

    @staticmethod
    def diff(arr):
        return pd.DataFrame(arr).diff()\
            .replace([np.inf, -np.inf], np.nan).ffill()\
            .fillna(0).values

    def _xform_data(self, indata):
        columns = []
        y_predict = indata[config.y_predict_column].values
        for k in helpers.tables:
            curr_indata = indata.rename(columns={
                k + '_last': 'close',
                k + '_high': 'high',
                k + '_low': 'low',
                k + '_volume': 'volume'
            })

            if BitcoinEnv.USE_INDICATORS:
                # Aren't these features besides close "feature engineering", which the neural-net should do away with?
                close = curr_indata['close'].values
                diff = np.diff(close)
                diff = np.insert(diff, 0, 0)
                sma15 = SMA(curr_indata, timeperiod=15)
                sma60 = SMA(curr_indata, timeperiod=60)
                rsi = RSI(curr_indata, timeperiod=14)
                atr = ATR(curr_indata, timeperiod=14)
                #columns += [close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr]
                columns += [close, diff, sma15, sma60, rsi, atr, curr_indata['volume'].values]
            else:
                columns += [
                    self.diff(curr_indata['close']),
                    self.diff(curr_indata['high']),
                    self.diff(curr_indata['low']),
                    self.diff(curr_indata['volume']),
                ]

        # --- Preprocess data
        xdata = np.nan_to_num(np.column_stack(columns))
        return xdata, y_predict

    def __str__(self):
        return 'BitcoinEnv'

    def close(self):
        pass

    def reset(self):
        self.time = time.time()
        self.cash = self.START_CAP
        self.value = self.START_CAP
        self.high_score = self.cash + self.value  # each score-breaker above initial capital = reward++
        start_timestep = 1  # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep

        # Fetch all rows from the database and & take random slice (based on limit).
        # TODO count(*) and fetch-random in SQL
        df = helpers.db_to_dataframe(scaler=None, limit=self.limit * 2)
        block_start = random.randint(0, len(df) - self.limit)
        block_end = block_start + self.limit
        df = df[block_start:block_end]
        self.x_train, self.y_train = self._xform_data(df)
        self.y_diff = self.pct_change(self.y_train)

        return np.append(self.x_train[start_timestep], [0., 0.])

    def execute(self, action):
        if self.continuous_actions:
            # signal = 0 if -40 < action < 5 else action
            signal = 0 if -5 < action < 5 else action
        else:
            signal = 5 if action == self.ACTION_BUY\
                else -5 if action == self.ACTION_SELL\
                else 0

        self.signals.append(signal)

        # (see prior commits for rewards using backtesting - pnl, cash, value)
        abs_sig = abs(signal)
        fee = 0  # 0.0025  # https://www.gdax.com/fees/BTC-USD
        reward = 0
        before = dict(cash=self.cash, value=self.value, total=self.cash+self.value)
        if signal > 0:
            if self.cash < abs_sig:
                pass
                # reward = -100
            else:
                self.value += abs_sig - abs_sig*fee
                self.cash -= abs_sig
        elif signal < 0:
            if self.value < abs_sig:
                pass
                # reward = -100
            else:
                self.cash += abs_sig - abs_sig*fee
                self.value -= abs_sig

        pct_change = self.y_diff[self.timestep + 1]  # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        self.value += pct_change * self.value
        total = self.value + self.cash
        if 'absolute' in self.name:
            reward += total
        else:
            reward += total - before['total']

        # Each time it sets a new high-score, extra reward
        if total > self.high_score:
            reward += total
            self.high_score = total

        self.timestep += 1
        # next_state = np.append(self.x_train[self.timestep], [self.cash, self.value])
        next_state = np.append(self.x_train[self.timestep], [
            self.cash,  # 0 if before['cash'] == 0 else (self.cash - before['cash']) / before['cash'],
            self.value  # 0 if before['value'] == 0 else (self.value - before['value']) / before['value'],
        ])

        terminal = int(self.timestep + 1 >= len(self.x_train))
        if terminal:
            self.time = round(time.time() - self.time)
            self.signals.append(0)  # Add one last signal (to match length)
            self.episode_cashs.append(self.cash)
            self.episode_values.append(self.value)
            self.action_counter = dict((round(k), v) for k, v in Counter(self.signals).most_common(5))
        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, reward, terminal

    @property
    def states(self):
        return dict(shape=(self.num_features(),), type='float')

    @property
    def actions(self):
        if self.continuous_actions:
            return dict(continuous=True, shape=(), min_value=-100, max_value=100)
        else:
            return dict(continuous=False, num_actions=3)  # BUY SELL HOLD

    def render(self): pass

    def seed(self, num):
        # TODO is all this correct/necessary?
        random.seed(num)
        np.random.seed(num)
        tf.set_random_seed(num)


    step = execute  # alias execute as step, called step by some frameworks

    def plotTrades(self, episode, reward, title='Results'):
        if hasattr(self, 'fig'):
            fig = self.fig
            plt.cla()
        else:
            fig = plt.figure(figsize=(25, 4))
        bt = Backtest(
            pd.Series(self.y_train),
            pd.Series(self.signals),
            signalType='capital'
        )
        pnl = bt.pnl.iloc[-1]
        bt.plotTrades()
        plt.suptitle('Ep:{} PNL:{} Cash:{} Value:{} Reward:{} '.format(
            episode,
            round(pnl, 1),
            round(self.cash, 1),
            round(self.value, 1),
            round(reward, 1))
        )
        if hasattr(self, 'fig'):
            fig.canvas.draw()
        else:
            block = bool(episode >= self.limit) # freeze on the last frame
            fig.canvas.set_window_title(title)
            plt.show(block=block)
            self.fig = fig