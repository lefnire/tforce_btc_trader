"""BTC trading environment. Trains on BTC price history to learn to buy/sell/hold.

This is an environment tailored towards TensorForce, not OpenAI Gym. Gym environments are
a standard used by many projects (Baselines, Coach, etc) and so would make sense to use; and TForce is compatible with
Gym envs. It's just that there's hoops to go through converting a Gym env to TForce, and it was ugly code. I actually
had it that way, you can search through Git if you want the Gym env; but one day I decided "I'm not having success with
any of these other projects, TForce is the best - I'm just gonna stick to that" and this approach was cleaner.

I actually do want to try NervanaSystems/Coach, that one's new since I started developing. Will require converting this
env back to Gym format. Anyone wanna give it a go?
"""

import random, time, requests, pdb, gdax, math, pickle, os, shutil, copy
from sklearn.model_selection import TimeSeriesSplit
from enum import Enum
import numpy as np
import pandas as pd
import talib.abstract as tlib
from box import Box
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from sklearn import preprocessing
from data.data import Data, Exchange, EXCHANGE
from utils import raise_refactor


class Mode(Enum):
    TRAIN = 'train'
    TEST = 'test'
    LIVE = 'live'
    TEST_LIVE = 'test_live'

# See 6fc4ed2 for Scaling states/rewards


class BitcoinEnv(Environment):
    EPISODE_LEN = 1000

    def __init__(self, hypers, cli_args={}):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.hypers = h = Box(hypers)
        self.cli_args = cli_args

        # cash/val start @ about $3.5k each. You should increase/decrease depending on how much you'll put into your
        # exchange accounts to trade with. Presumably the agent will learn to work with what you've got (cash/value
        # are state inputs); but starting capital does effect the learning process.
        self.start_cash, self.start_value = 5., 5.  # .4, .4

        # We have these "accumulator" objects, which collect values over steps, over episodes, etc. Easier to keep
        # same-named variables separate this way.
        acc = dict(
            ep=dict(
                i=-1,  # +1 in reset, makes 0
                returns=[],
                uniques=[],
            ),
            step=dict(),  # setup in reset()
        )
        self.acc = Box(train=copy.deepcopy(acc), test=copy.deepcopy(acc))
        self.data = Data(ep_len=self.EPISODE_LEN, arbitrage=h.custom.arbitrage, indicators={})

        # gdax min order size = .01btc; kraken = .002btc
        self.min_trade = {Exchange.GDAX: .01, Exchange.KRAKEN: .002}[EXCHANGE]
        self.update_btc_price()

        # Action space
        # see {last_good_commit_ for action_types other than 'single_discrete'
        # In single_discrete, we allow buy2%, sell2%, hold (and nothing else)
        self.actions_ = dict(type='int', shape=(), num_actions=3)

        # Observation space
        # width = step-window (150 time-steps)
        # height = nothing (1)
        # channels = features/inputs (price actions, OHCLV, etc).
        self.cols_ = self.data.df.shape[1]
        shape = (h.custom.net.step_window, 1, self.cols_)
        self.states_ = dict(type='float', shape=shape)

    def __str__(self): return 'BitcoinEnv'

    def close(self): pass

    @property
    def states(self): return self.states_

    @property
    def actions(self): return self.actions_

    # We don't want random-seeding for reproducibilityy! We _want_ two runs to give different results, because we only
    # trust the hyper combo which consistently gives positive results.
    def seed(self, seed=None): return

    def update_btc_price(self):
        self.btc_price = 8000
        # try:
        #     self.btc_price = int(requests.get(f"https://api.cryptowat.ch/markets/{EXCHANGE.value}/btcusd/price").json()['result']['price'])
        # except:
        #     self.btc_price = self.btc_price or 8000

    def xform_data(self, df):
        # TODO here was autoencoder, talib indicators, price-anchoring
        raise_refactor()

    def get_next_state(self):
        acc = self.acc[self.mode.value]
        X, _ = self.data.get_data(acc.ep.i, acc.step.i)
        return X.values[:, np.newaxis, :]  # height, width(nothing), depth

    def reset(self):
        acc = self.acc[self.mode.value]
        acc.step.i = 0
        acc.step.cash, acc.step.value = self.start_cash, self.start_value
        acc.step.hold_value = self.start_value
        acc.step.totals = Box(
            trade=[self.start_cash + self.start_value],
            hold=[self.start_cash + self.start_value]
        )
        acc.step.signals = []
        if self.mode == Mode.TEST:
            acc.ep.i = self.acc.train.ep.i + 1
        elif self.mode == Mode.TRAIN:
            acc.ep.i += 1

        self.data.reset_cash_val()
        self.data.set_cash_val(acc.ep.i, acc.step.i, 0., 0.)
        return self.get_next_state()

    def execute(self, action):
        acc = self.acc[self.mode.value]
        totals = acc.step.totals
        h = self.hypers

        act_pct = {
            0: -.02,
            1: 0,
            2: .02
        }[action]
        act_btc = act_pct * (acc.step.cash if act_pct > 0 else acc.step.value)

        fee = {
            Exchange.GDAX: 0.0025,  # https://support.gdax.com/customer/en/portal/articles/2425097-what-are-the-fees-on-gdax-
            Exchange.KRAKEN: 0.0026  # https://www.kraken.com/en-us/help/fees
        }[EXCHANGE]

        # Perform the trade. In training mode, we'll let it dip into negative here, but then kill and punish below.
        # In testing/live, we'll just block the trade if they can't afford it
        if act_pct > 0:
            if acc.step.cash < self.min_trade:
                act_btc = -(self.start_cash + self.start_value)
            elif act_btc < self.min_trade:
                act_btc = 0
            else:
                acc.step.value += act_btc - act_btc*fee
            acc.step.cash -= act_btc

        elif act_pct < 0:
            if acc.step.value < self.min_trade:
                act_btc = -(self.start_cash + self.start_value)
            elif abs(act_btc) < self.min_trade:
                act_btc = 0
            else:
                acc.step.cash += abs(act_btc) - abs(act_btc)*fee
            acc.step.value -= abs(act_btc)

        acc.step.signals.append(float(act_btc))  # clipped signal
        # acc.step.signals.append(np.sign(act_pct))  # indicates an attempted trade

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        # pct_change = self.prices_diff[acc.step.i + 1]
        _, y = self.data.get_data(acc.ep.i, acc.step.i)  # TODO verify
        pct_change = y[self.data.target]

        acc.step.value += pct_change * acc.step.value
        total_now = acc.step.value + acc.step.cash
        totals.trade.append(total_now)

        # calculate what the reward would be "if I held", to calculate the actual reward's _advantage_ over holding
        hold_before = acc.step.hold_value
        acc.step.hold_value += pct_change * hold_before
        totals.hold.append(acc.step.hold_value + self.start_cash)

        reward = 0

        acc.step.i += 1

        self.data.set_cash_val(
            acc.ep.i, acc.step.i,
            acc.step.cash/self.start_cash,
            acc.step.value/self.start_value
        )
        next_state = self.get_next_state()

        terminal = int(acc.step.i + 1 >= self.EPISODE_LEN)
        if acc.step.value < 0 or acc.step.cash < 0:
            terminal = True
        if terminal and self.mode in (Mode.TRAIN, Mode.TEST):
            # We're done.
            acc.step.signals.append(0)  # Add one last signal (to match length)
            reward = self.get_return()
            if np.unique(acc.step.signals).shape[0] == 1:
                reward = -(self.start_cash + self.start_value)  # slam if you don't do anything

        if terminal and self.mode in (Mode.LIVE, Mode.TEST_LIVE):
            raise_refactor()

        # if acc.step.value <= 0 or acc.step.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def get_return(self, adv=True):
        acc = self.acc[self.mode.value]
        totals = acc.step.totals
        trade = (totals.trade[-1] / totals.trade[0] - 1)
        hold = (totals.hold[-1] / totals.hold[0] - 1)
        return trade - hold if adv else trade

    def episode_finished(self, runner):
        if self.mode == Mode.TRAIN: return True

        acc = self.acc.test
        totals = acc.step.totals
        signals = np.array(acc.step.signals)
        n_uniques = np.unique(signals).shape[0]
        ret = self.get_return()
        hold_ret = totals.hold[-1] / totals.hold[0] - 1

        acc.ep.returns.append(float(ret))
        acc.ep.uniques.append(n_uniques)

        # Print (limit to note-worthy)
        lt_0 = (signals < 0).sum()
        eq_0 = (signals == 0).sum()
        gt_0 = (signals > 0).sum()
        completion = int(acc.ep.i * self.data.ep_stride / self.data.df.shape[0] * 100)
        steps = f"\tSteps: {acc.step.i}"

        fm = '%.3f'
        print(f"{completion}%{steps}\tTrade: {fm%ret}\tHold: {fm%hold_ret}\tTrades:\t{lt_0}[<0]\t{eq_0}[=0]\t{gt_0}[>0]")
        return True

    def run_deterministic(self, runner, print_results=True):
        next_state, terminal = self.reset(), False
        while not terminal:
            next_state, terminal, reward = self.execute(runner.agent.act(next_state, deterministic=True, independent=True))
        if print_results: self.episode_finished(None)

    def train_and_test(self, agent):
        runner = Runner(agent=agent, environment=self)
        train_steps = 20000  # TODO something self.data.df.shape[0]... self.EPISODE_LEN...

        try:
            while self.data.has_more(self.acc.train.ep.i):
                self.mode = Mode.TRAIN
                # max_episode_timesteps not required, since we kill on (cash|value)<0 or max_repeats
                runner.run(timesteps=train_steps)
                self.mode = Mode.TEST
                self.run_deterministic(runner, print_results=True)
        except IndexError:
            # FIXME data.has_more() issues
            pass
        except KeyboardInterrupt:
            # Lets us kill training with Ctrl-C and skip straight to the final test. This is useful in case you're
            # keeping an eye on terminal and see "there! right there, stop you found it!" (where early_stop & n_steps
            # are the more methodical approaches)
            print('Keyboard interupt, killing training')
            pass

    def run_live(self, agent, test=True):
        raise_refactor()
