import random
import math
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
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    def __init__(self, use_indicators=False, limit=1000):
        """Initializes a minimal test environment."""
        df = helpers.db_to_dataframe(scaler=preprocessing.MinMaxScaler)

        # We're going to fetch all rows from the database and store them. We'll take random slices (based on limit)
        # on each reset, setting self.x_train self.y_train etc to those slices. Alternatively we could fetch the random
        # block from the database (limit,start) each reset()
        self.x_all, self.y_all = self._xform_data(df, use_indicators)
        self.y_all_diff = pd.Series(self.y_all).pct_change()\
            .replace([np.inf, -np.inf, np.nan], np.nan).ffill()  # TODO is this the right thinking?
            # .replace([np.inf, -np.inf, np.nan], [0.9, -0.9, 0.])
        self.num_features = (7 if use_indicators else 4) * len(helpers.tables)
        self.limit = limit

    def _xform_data(self, indata, use_indicators):
        columns = []
        y_predict = indata[config.data.y_predict_column].values
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
        self.cash = 100; self.value = 100

        block_start = random.randint(0, len(self.y_all) - self.limit)
        block_end = block_start + self.limit
        self.x_train = self.x_all[block_start:block_end]
        self.y_train = self.y_all[block_start:block_end]
        self.y_diff = self.y_all_diff[block_start:block_end]

        start_timestep = 2 # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        return self.x_train[start_timestep]

    def execute(self, action):
        signal = 5 if action == self.ACTION_BUY\
            else -40 if action == self.ACTION_SELL\
            else 0
        self.signals.append(signal)

        ## Attempt 1
        # bt = Backtest(
        #     pd.Series([y for y in self.y_train[self.timestep-1:self.timestep+1]]),
        #     pd.Series(self.signals[self.timestep-1:self.timestep+1]),
        #     signalType='capital'
        # )
        # perc_change = pd.Series(list(self.y_train[self.timestep-1:self.timestep+1])).pct_change()[1]
        # reward = perc_change * signal

        ## Attempt 2
        # bt = Backtest(
        #     pd.Series(self.y_train[:self.timestep]),
        #     pd.Series(self.signals[:self.timestep]),
        #     signalType='capital'
        # )
        # # Fixme sometimes pnl is inf/-inf/nan - stop using TWP? see https://github.com/mementum/backtrader & others
        # reward = np.diff(bt.pnl.iloc[-2:])[-1]
        # # if reward == float('Inf') or reward == float('-Inf') or math.isnan(reward):
        # #     print('reward=nan')

        ## Attempt 3
        b4_cash, b4_val = self.cash, self.value
        abs_sig = abs(signal)
        fee = 0.0025 # TODO verify https://www.gdax.com/fees/BTC-USD
        if signal > 0:
            self.value += abs_sig - abs_sig*fee
            self.cash -= abs_sig
        elif signal < 0:
            self.cash += abs_sig - abs_sig*fee
            self.value -= abs_sig

        pct_change = self.y_diff.iloc[self.timestep + 1] # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        self.value += pct_change * self.value
        # reward = (self.value + self.cash) - (b4_cash + b4_val)
        reward = self.value + self.cash
        if self.value <= 0 or self.cash <= 0:
            # we can't go negative in real life, so _extra_ punishment
            reward -= 100

        self.timestep += 1
        next_state = self.x_train[self.timestep]
        terminal = int(self.timestep+1 >= len(self.x_train))
        if terminal:
            self.signals.append(0)  # Add one last signal (to match length)
        if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, reward, terminal

    @property
    def states(self):
        return dict(shape=(self.num_features,), type='float')

    @property
    def actions(self):
        return dict(continuous=False, num_actions=3) # BUY SELL HOLD
        # return dict(continuous=True, num_actions=(), min_value=-100, max_value=100) # BUY SELL HOLD

    def plotTrades(self, episode, reward):
        if hasattr(self, 'fig'):
            fig = self.fig
            plt.cla()
        else:
            fig = plt.figure(figsize=(12, 10))
        bt = Backtest(
            pd.Series(self.y_train),
            pd.Series(self.signals),
            signalType='capital'
        )
        pnl = bt.pnl.iloc[-1]
        bt.plotTrades()
        plt.suptitle('Episode=%i PNL=$%2f "Reward"=%2f' % (episode, pnl, reward))
        if hasattr(self, 'fig'):
            fig.canvas.draw()
        else:
            block = bool(episode >= self.limit) # freeze on the last frame
            plt.show(block=block)
            self.fig = fig