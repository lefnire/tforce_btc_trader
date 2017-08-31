import random, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talib.abstract import SMA, RSI, ATR
from sklearn import preprocessing
from tradingWithPython.lib.backtest import Backtest

from tensorforce import util, TensorForceError
from tensorforce.environments import Environment

import helpers
from helpers import config


class BitcoinEnv(Environment):
    ACTION_BUY1 = 0
    ACTION_BUY2 = 1
    ACTION_SELL = 2
    ACTION_HOLD = 3

    def __init__(self, use_indicators=False, limit=1000, agent_type='DQNAgent'):
        """Initializes a minimal test environment."""

        # limit here is > self.limit since we want bit window (helpers.limit) to random-choose (self.limit)
        df = helpers.db_to_dataframe(scaler=None, limit=limit*2)

        # We're going to fetch all rows from the database and store them. We'll take random slices (based on limit)
        # on each reset, setting self.x_train self.y_train etc to those slices. Alternatively we could fetch the random
        # block from the database (limit,start) each reset()
        self.x_all, self.y_all = self._xform_data(df, use_indicators)
        self.y_all_diff = pd.Series(self.y_all).pct_change()\
            .replace([np.inf, -np.inf, np.nan], [0.9, -0.9, 0.])
        self.num_features = (7 if use_indicators else 4) * len(helpers.tables) +2 # for cash/value
        self.limit = limit
        self.agent_type = agent_type
        self.episode_cashs = []; self.episode_values = []
        self.name = None

    def _xform_data(self, indata, use_indicators):
        columns = []
        y_predict = indata[config.y_predict_column].values
        for k in helpers.tables:
            curr_indata = indata.rename(columns={
                k + '_last': 'close',
                k + '_high': 'high',
                k + '_low': 'low',
                k + '_volume': 'volume'
            })

            if use_indicators:
                # Aren't these features besides close "feature engineering", which the neural-net should do away with?
                close = curr_indata['close'].values
                diff = np.diff(close)
                diff = np.insert(diff, 0, 0)
                sma15 = SMA(curr_indata, timeperiod=15)
                sma60 = SMA(curr_indata, timeperiod=60)
                rsi = RSI(curr_indata, timeperiod=14)
                atr = ATR(curr_indata, timeperiod=14)
                columns += [close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr]
            else:
                columns += [
                    curr_indata['close'].values,
                    curr_indata['high'].values,
                    curr_indata['low'].values,
                    curr_indata['volume'].values
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
        self.cash = 100; self.value = 100

        block_start = random.randint(0, len(self.y_all) - self.limit)
        block_end = block_start + self.limit
        self.x_train = self.x_all[block_start:block_end]
        self.y_train = self.y_all[block_start:block_end]
        self.y_diff = self.y_all_diff[block_start:block_end]

        start_timestep = 2 # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        return np.append(self.x_train[start_timestep], [self.cash, self.value])

    def execute(self, action):
        if self.agent_type in ['DQNAgent']:
            signal = 5 if action == self.ACTION_BUY1\
                else 40 if action == self.ACTION_BUY2\
                else -40 if action == self.ACTION_SELL\
                else 0
        elif self.agent_type in ['PPOAgent']:
            signal = 0 if -40 < action < 5 else action
        elif self.agent_type in ['NAFAgent']:
            # doesn't do min_value max_value!
            # if action < 0 or action > 1: print(action)
            signal = (action - .5) * 200

        self.signals.append(signal)

        # (see prior commits for rewards using backtesting - pnl, cash, value)
        abs_sig = abs(signal)
        fee = 0.0025  # https://www.gdax.com/fees/BTC-USD
        reward = 0
        cashb4, valueb4 = self.cash, self.value
        if signal > 0:
            if self.cash < abs_sig:
                reward = -1000
            self.value += abs_sig - abs_sig*fee
            self.cash -= abs_sig
        elif signal < 0:
            if self.value < abs_sig:
                reward = -1000
            self.cash += abs_sig - abs_sig*fee
            self.value -= abs_sig

        pct_change = self.y_diff.iloc[self.timestep + 1]  # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        self.value += pct_change * self.value
        if 'absolute' in self.name:
            reward += self.value + self.cash
        else:
            reward += (self.value + self.cash) - (cashb4 + valueb4)

        self.timestep += 1
        next_state = np.append(self.x_train[self.timestep], [self.cash, self.value])
        terminal = int(self.timestep+1 >= len(self.x_train))
        if terminal:
            self.time = round(time.time() - self.time)
            self.signals.append(0)  # Add one last signal (to match length)
            self.episode_cashs.append(self.cash)
            self.episode_values.append(self.value)
        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, reward, terminal

    @property
    def states(self):
        return dict(shape=(self.num_features,), type='float')

    @property
    def actions(self):
        if self.agent_type == 'DQNAgent':
            return dict(continuous=False, num_actions=4)  # BUY1 BUY2 SELL HOLD
        elif self.agent_type in ['PPOAgent', 'NAFAgent']:
            return dict(continuous=True, shape=(), min_value=-100, max_value=100)

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