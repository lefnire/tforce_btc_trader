import random, time, re, math
import numpy as np
import pandas as pd
from talib.abstract import SMA, RSI, ATR, EMA
from collections import Counter
from sqlalchemy.sql import text
import tensorflow as tf
from box import Box
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from sklearn.preprocessing import RobustScaler, robust_scale
import pdb

ALLOW_SEED = False
TIMESTEPS = int(2e6)
START_CAP = 1500
WINDOW = 150

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
        self.acc = Box(
            episode=dict(
                i=0,
                total_steps=0,
                advantages=[],
                uniques=[]
            ),
            step=dict(i=0)  # setup in reset()
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
        # Note: don't scale/normalize here, since we'll normalize w/ self.price/step_acc.cash after each action
        return states, prices

    def _reshape_window_for_conv2d(self, window):
        if len(data.tables) == 1:
            return np.expand_dims(window, -1)
        elif len(data.tables) == 2:  # default (risk arbitrage)
            return np.transpose([window[:, 0:self.cols_], window[:, self.cols_:]], (1, 2, 0))
        else:
            raise NotImplementedError('TODO Implement conv2d features depth > 2')

    def set_testing(self, testing):
        """Make sure to call this before reset()!"""
        print("Testing" if testing else "Training")
        self.testing = testing
        self.row_ct = data.count_rows(self.conn, arbitrage=self.hypers.arbitrage)
        split = .9
        n_train, n_test = int(self.row_ct * split), int(self.row_ct * (1 - split))
        limit, offset = (n_test, n_train) if testing else (n_train, 0)
        df = data.db_to_dataframe(self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage)

        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._diff(self.prices, percent=True)

    def reset(self):
        self.time = time.time()
        step_acc, ep_acc = self.acc.step, self.acc.episode
        # Cash & value are the real scores - how much we end up with at the end of an episode
        step_acc.cash = step_acc.value = START_CAP
        # But for our purposes, we care more about "how much better is what we made than if we held". We're training
        # a trading bot, not an investing bot. So we compare these at the end, calling it "advantage"
        step_acc.hold = Box(value=START_CAP, cash=START_CAP)
        start_timestep = WINDOW if self.conv2d else 1  # advance some steps just for cushion, various operations compare back a couple steps
        step_acc.i = start_timestep
        step_acc.signals = [0] * start_timestep
        step_acc.repeats = 1
        ep_acc.i += 1

        if self.conv2d:
            window = self.observations[start_timestep - WINDOW:start_timestep]
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

        step_acc, ep_acc = self.acc.step, self.acc.episode

        step_acc.signals.append(float(signal))

        fee = 0.0026  # https://www.kraken.com/en-us/help/fees
        reward = 0
        abs_sig = abs(signal)
        before = Box(cash=step_acc.cash, value=step_acc.value, total=step_acc.cash+step_acc.value)
        if signal > 0:
            step_acc.value += abs_sig - abs_sig*fee
            step_acc.cash -= abs_sig
        elif signal < 0:
            step_acc.cash += abs_sig - abs_sig*fee
            step_acc.value -= abs_sig

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        diff_loc = step_acc.i if self.conv2d else step_acc.i + 1
        pct_change = self.prices_diff[diff_loc]
        step_acc.value += pct_change * step_acc.value
        total = step_acc.value + step_acc.cash
        reward += total - before.total

        # calculate what the reward would be "if I held", to calculate the actual reward's _advantage_ over holding
        before = step_acc.hold
        before.value += pct_change * before.value

        # Encourage diverse behavior. hypers.punish_repeats method means punishing homogenous behavior
        recent_actions = np.array(step_acc.signals[-step_acc.repeats:])
        if np.any(recent_actions > 0) and np.any(recent_actions < 0) and np.any(recent_actions == 0):
            step_acc.repeats = 1  # reset penalty-growth
        else:
            if self.hypers.punish_repeats:
                # We want a trade every x minutes. Increase punishment over time until it's 1 at that step
                # reward -= self.scaler.avg_reward() * (step_acc.repeats/(60*10))
                n_minutes = 2
                reward -= step_acc.repeats / (60 * n_minutes)
            step_acc.repeats += 1  # grow the penalty with time

        step_acc.i += 1
        ep_acc.total_steps += 1
        # 0 if before['cash'] == 0 else (step_acc.cash - before['cash']) / before['cash'],
        # 0 if before['value'] == 0 else (step_acc.value - before['value']) / before['value'],
        cash_scaled, val_scaled = step_acc.cash / START_CAP,  step_acc.value / START_CAP
        if self.conv2d:
            window = self.observations[step_acc.i - WINDOW:step_acc.i]
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
        # Kill and punish if (a) agent ran out of money; (b) is doing nothing for way too long
        if step_acc.cash < 0 or step_acc.value < 0 or step_acc.repeats >= 20000:
            reward -= 1000
            terminal = True
        if terminal:
            step_acc.signals.append(0)  # Add one last signal (to match length)

        # if step_acc.value <= 0 or step_acc.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def episode_finished(self, runner):
        step_acc, ep_acc = self.acc.step, self.acc.episode
        time_ = round(time.time() - self.time)
        signals = step_acc.signals

        advantage = ((step_acc.cash + step_acc.value) - START_CAP * 2) - \
                    ((step_acc.hold.value + step_acc.hold.cash) - START_CAP * 2)
        self.acc.episode.advantages.append(advantage)
        self.acc.episode.uniques.append(float(len(np.unique(signals))))

        # Print (limit to note-worthy)
        common = dict((round(k), v) for k, v in Counter(signals).most_common(5))
        high, low = max(signals), min(signals)
        completion = f"|{int(ep_acc.total_steps / TIMESTEPS * 100)}%"
        print(f"{ep_acc.i}|⌛:{step_acc.i}{completion}\tA:{'%.3f'%advantage}\t{common}(high:{'%.2f'%high},low:{'%.2f'%low})")
        return True

    def train_and_test(self, agent):
        timesteps, n_tests = int(2e6), 40
        n_train = timesteps // n_tests
        i = 0

        runner = Runner(agent=agent, environment=self)
        self.set_testing(False)
        while i <= n_tests:
            runner.run(timesteps=n_train, max_episode_timesteps=n_train)
            runner.run(deterministic=True, episode_finished=self.episode_finished, episodes=1)
            i += 1
        # Last test is the biggie (test set)
        self.set_testing(True)
        runner.run(deterministic=True, episode_finished=self.episode_finished, episodes=1)
