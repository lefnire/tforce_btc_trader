"""BTC trading environment. Trains on BTC price history to learn to buy/sell/hold.

This is an environment tailored towards TensorForce, not OpenAI Gym. Gym environments are
a standard used by many projects (Baselines, Coach, etc) and so would make sense to use; and TForce is compatible with
Gym envs. It's just that there's hoops to go through converting a Gym env to TForce, and it was ugly code. I actually
had it that way, you can search through Git if you want the Gym env; but one day I decided "I'm not having success with
any of these other projects, TForce is the best - I'm just gonna stick to that" and this approach was cleaner.

I actually do want to try NervanaSystems/Coach, that one's new since I started developing. Will require converting this
env back to Gym format. Anyone wanna give it a go?
"""

import random, time, requests, pdb, gdax, math
from enum import Enum
import numpy as np
import pandas as pd
import talib.abstract as tlib
from collections import Counter
import tensorflow as tf
from box import Box
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from sklearn.preprocessing import RobustScaler, robust_scale
from data.data import Exchange, EXCHANGE
from data import data
from autoencoder import AutoEncoder


class Mode(Enum):
    TRAIN = 1
    TEST = 2
    LIVE = 3
    TEST_LIVE = 4


class Scaler(object):
    """If we have `hypers.scale=True`, we use this class to scale everything (price-actions, rewards, etc). Using this
    instead of TForce's built-in preprocessing (http://tensorforce.readthedocs.io/en/latest/preprocessing.html) since
    this gives more flexibility, but it's basically the same thing. Someone may want to check me on that statement by
    reading those docs and trying TForce's preprocessing instead of this.

    One important bit here is the use of RobustScaler with a quantile_range. This allows us to handle outliers, which
    abound in the data. Sometimes we have a timeseries hole, and suddenly we're up a billion percent. Sometimes whales
    pump-and-dump to screw with the market. RobustScaler lets us "ignore" those moments.

    TODO someone will want to double-check my work on this scaling approach in general. Best of my knowledges, but I'm
    a newb.
    """

    # At some point we can safely say "I've seen enough, just scale (don't fit) going forward")
    STOP_AT = int(1e6)
    SKIP = 15

    # state types
    REWARD = 1
    SERIES = 2
    STATIONARY = 3

    def __init__(self):
        self.scalers = {
            self.REWARD: RobustScaler(quantile_range=(5., 95.)),
            self.STATIONARY: RobustScaler(quantile_range=(5., 95.))
        }
        self.data = {
            self.REWARD: [],
            self.STATIONARY: []
        }
        self.done = False
        self.i = 0

    def _should_skip(self):
        # After we've fitted enough (see STOP_AT), start returning direct-transforms for performance improvement
        # Skip every few fittings. Each individual doesn't contribute a whole lot anyway, and costs a lot
        return self.done or (self.i % self.SKIP != 0 and self.i > self.SKIP)

    def transform(self, input, kind, force=False):
        # this is awkward; we only want to increment once per step, but we're calling this fn 3x per step (once
        # for series, once for stationary, once for reward). Explicitly saying "only increment for one of those" here.
        # Using STATIONARY since SERIES might be called once per timestep in a conv window. TODO Seriously awkward
        if kind == self.STATIONARY: self.i += 1

        scaler = self.scalers[kind]
        matrix = np.array(input).ndim == 2
        if self._should_skip() and not force:
            if matrix: return scaler.transform(input)
            return scaler.transform([input])[-1]
        # Fit, transform, return
        data = self.data[kind]
        if matrix:
            self.data[kind] += input.tolist()
            ret = scaler.fit_transform(data)[-input.shape[0]:]
        else:
            data.append(input)
            ret = scaler.fit_transform(data)[-1]
        if self.i >= self.STOP_AT and not self.done:
            self.done = True
            del self.data  # Clear up memory, fitted scalers have all the info we need.
        return ret


# keep this globally around for all runs forever
scalers = {}


class BitcoinEnv(Environment):
    EPISODE_LEN = 5000

    def __init__(self, hypers, cli_args={}):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.hypers = h = Box(hypers)
        self.conv2d = self.hypers['net.type'] == 'conv2d'
        self.all_or_none = self.hypers.action_type == 'all_or_none'
        self.cli_args = cli_args

        # cash/val start @ about $3.5k each. You should increase/decrease depending on how much you'll put into your
        # exchange accounts to trade with. Presumably the agent will learn to work with what you've got (cash/value
        # are state inputs); but starting capital does effect the learning process.
        self.start_cash, self.start_value = .4, .4

        # We have these "accumulator" objects, which collect values over steps, over episodes, etc. Easier to keep
        # same-named variables separate this way.
        self.acc = Box(
            episode=dict(
                i=0,
                total_steps=0,
                sharpes=[],
                returns=[],
                uniques=[]
            ),
            step=dict(),  # setup in reset()
            tests=dict(
                i=0,
                n_tests=0
            )
        )
        self.mode = Mode.TRAIN
        self.conn = data.engine.connect()

        # gdax min order size = .01btc; kraken = .002btc
        self.min_trade = {Exchange.GDAX: .01, Exchange.KRAKEN: .002}[EXCHANGE]
        self.update_btc_price()

        # Should be one scaler for any permutation of data (since the columns need to align exactly)
        scaler_k = f'{h.arbitrage}|{h.indicators_count}|{h.repeat_last_state}|{h.action_type}'
        if scaler_k not in scalers:
            scalers[scaler_k] = Scaler()
        self.scaler = scalers[scaler_k]

        # Our data is too high-dimensional for the way MemoryModel handles batched episodes. Reduce it (don't like this)
        all_data = data.db_to_dataframe(self.conn, arbitrage=h.arbitrage)
        self.all_observations, self.all_prices = self.xform_data(all_data)
        self.all_prices_diff = self.diff(self.all_prices, percent=True)

        # Calculate a possible reward to be used as an average for repeat-punishing
        self.possible_reward = self.start_value * np.median([p for p in self.all_prices_diff if p > 0])
        print('possible_reward', self.possible_reward)

        # Action space
        trade_cap = self.min_trade * 2  # not necessary to limit it like this, doing for my own sanity in live-mode
        if h.action_type == 'single':
            # In single_action we discard any vals b/w [-min_trade, +min_trade] and call it "hold" (in execute())
            self.actions_ = dict(type='float', shape=(), min_value=-trade_cap, max_value=trade_cap)
        elif h.action_type == 'multi':
            # In multi-modal, hold is an actual action (in which case we discard "amount")
            self.actions_ = dict(
                action=dict(type='int', shape=(), num_actions=3),
                amount=dict(type='float', shape=(), min_value=self.min_trade, max_value=trade_cap))
        elif h.action_type == 'all_or_none':
            self.actions_ = dict(type='int', shape=(), num_actions=3)

        # Observation space
        stationary_ct = 1 if self.all_or_none else 2
        self.cols_ = self.all_observations.shape[1]
        self.states_ = dict(
            series=dict(type='float', shape=self.cols_),  # all state values that are time-ish
            stationary=dict(type='float', shape=stationary_ct)  # everything that doesn't care about time
        )

        if self.conv2d:
            # width = step-window (150 time-steps)
            # height = nothing (1)
            # channels = features/inputs (price actions, OHCLV, etc).
            self.states_['series']['shape'] = (h.step_window, 1, self.cols_)
            if h.repeat_last_state:
                self.states_['stationary']['shape'] += self.cols_

    def __str__(self): return 'BitcoinEnv'

    def close(self): self.conn.close()

    @property
    def states(self): return self.states_

    @property
    def actions(self): return self.actions_

    # We don't want random-seeding for reproducibilityy! We _want_ two runs to give different results, because we only
    # trust the hyper combo which consistently gives positive results.
    def seed(self, seed=None): return

    def update_btc_price(self):
        try:
            self.btc_price = int(requests.get(f"https://api.cryptowat.ch/markets/{EXCHANGE.value}/btcusd/price").json()['result']['price'])
        except:
            self.btc_price = self.btc_price or 8000

    def diff(self, arr, percent=False):
        series = pd.Series(arr)
        diff = series.pct_change() if percent else series.diff()
        diff.iloc[0] = 0  # always NaN, nothing to compare to

        # Remove outliers (turn them to NaN)
        q = diff.quantile(0.99)
        diff = diff.mask(diff > q, np.nan)

        # then forward-fill the NaNs.
        return diff.replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    def xform_data(self, df):
        columns = []
        ind_ct = self.hypers.indicators_count
        tables_ = data.get_tables(self.hypers.arbitrage)
        percent = self.hypers.pct_change
        for table in tables_:
            name, cols, ohlcv = table['name'], table['cols'], table.get('ohlcv', {})
            columns += [self.diff(df[f'{name}_{k}'], percent) for k in cols]

            # Add extra indicator columns
            if ohlcv and ind_ct:
                ind = pd.DataFrame()
                # TA-Lib requires specifically-named columns (OHLCV)
                for k, v in ohlcv.items():
                    ind[k] = df[f"{name}_{v}"]

                # Sort these by effectiveness. I'm no expert, so if this seems off please submit a PR! Later after
                # you've optimized the other hypers, come back here and create a hyper for every indicator you want to
                # try (zoom in on indicators)
                best_indicators = [
                    tlib.MOM,
                    tlib.SMA,
                    # tlib.BBANDS,  # TODO signature different; special handling
                    tlib.RSI,
                    tlib.EMA,
                    tlib.ATR
                ]
                for i in range(ind_ct):
                    columns += [self.diff(best_indicators[i](ind, timeperiod=self.hypers.indicators_window), percent)]

        states = np.column_stack(columns)
        prices = df[data.target].values

        # Remove padding at the start of all data. Indicators are aggregate fns, so don't count until we have
        # that much historical data
        if ind_ct:
            states = states[self.hypers.indicators_window:]
            prices = prices[self.hypers.indicators_window:]

        # Pre-scale all price actions up-front, since they don't change. We'll scale changing values real-time elsewhere
        if self.hypers.scale:
            states = robust_scale(states, quantile_range=(5., 95.))

        # Reducing the dimensionality of our states (OHLCV + indicators + arbitrage => 5 or 6 weights)
        # because TensorForce's memory branch changed Policy Gradient models' batching from timesteps to episodes.
        # This takes of way too much GPU RAM for us, so we had to cut back in quite a few areas (num steps to train
        # per episode, episode batch_size, and especially states:
        if self.cli_args.autoencode:
            ae = AutoEncoder()
            states = ae.fit_transform_tied(states)

        return states, prices

    def use_dataset(self, mode, full_set=False):
        """Fetches, transforms, and stores the portion of data you'll be working with (ie, 80% train data, 20% test
        data, or the live database). Make sure to call this before reset()!
        """
        self.mode = mode
        if mode in (Mode.LIVE, Mode.TEST_LIVE):
            self.conn = data.engine_live.connect()
            # Work with 6000 timesteps up until the present (play w/ diff numbers, depends on LSTM)
            # Offset=0 data.py currently pulls recent-to-oldest, then reverses
            rampup = int(1e5)  # 6000  # FIXME temporarily using big number to build up Scaler (since it's not saved)
            limit, offset = (rampup, 0) # if not self.conv2d else (self.hypers.step_window + 1, 0)
            df, self.last_timestamp = data.db_to_dataframe(
                self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage, last_timestamp=True)
            # save away for now so we can keep transforming it as we add new data (find a more efficient way)
            self.df = df
        else:
            row_ct = data.count_rows(self.conn, arbitrage=self.hypers.arbitrage)
            split = .9  # Using 90% training data.
            n_train, n_test = int(row_ct * split), int(row_ct * (1 - split))
            if mode == mode.TEST:
                offset = n_train
                limit = 40000 if full_set else 10000  # should be `n_test` in full_set, getting idx errors
            else:
                # Grab a random window from the 90% training data. The random bit is important so the agent
                # sees a variety of data. The window-size bit is a hack: as long as the agent doesn't die (doesn't cause
                # `terminal=True`), PPO's MemoryModel can keep filling up until it crashes TensorFlow. This ensures
                # there's a stopping point (limit). I'd rather see how far he can get w/o dying, figure out a solution.
                limit = self.EPISODE_LEN
                offset_start = 0 if not self.conv2d else self.hypers.step_window + 1
                offset = random.randint(offset_start, n_train - self.EPISODE_LEN)

        self.offset, self.limit = offset, limit
        self.prices = self.all_prices[offset:offset+limit]
        self.prices_diff = self.all_prices_diff[offset:offset+limit]

    def get_next_state(self, i, stationary):
        i = i + self.offset
        series = self.all_observations[i]
        if self.hypers.scale:
            # series already scaled in self._xform_data()
            stationary = self.scaler.transform(stationary, Scaler.STATIONARY).tolist()

        if self.conv2d:
            if self.hypers.repeat_last_state:
                stationary += series.tolist()
            # Take note of the +1 here. LSTM uses a single index [i], which grabs the list's end. Conv uses a window,
            # [-something:i], which _excludes_ the list's end (due to Python indexing). Without this +1, conv would
            # have a 1-step-behind delayed response.
            window = self.all_observations[i - self.hypers.step_window + 1:i + 1]
            series = np.expand_dims(window, axis=1)
        return dict(series=series, stationary=stationary)

    def reset(self):
        step_acc, ep_acc = self.acc.step, self.acc.episode
        step_acc.i = 0
        step_acc.cash, step_acc.value = self.start_cash, self.start_value
        step_acc.hold_value = self.start_value
        step_acc.totals = Box(
            trade=[self.start_cash + self.start_value],
            hold=[self.start_cash + self.start_value]
        )
        step_acc.signals = []
        if self.all_or_none:
            step_acc.last_action = 1
        ep_acc.i += 1

        stationary = [step_acc.last_action] if self.all_or_none else [self.start_cash, self.start_value]
        return self.get_next_state(0, stationary)

    def execute(self, actions):
        step_acc, ep_acc = self.acc.step, self.acc.episode
        h = self.hypers

        if h.action_type == 'single':
            signal = 0 if -self.min_trade < actions < self.min_trade else actions
        elif h.action_type == 'multi':
            # Two actions: `action` (buy/sell/hold) and `amount` (how much)
            signal = {
                0: -1,  # make amount negative
                1: 0,  # hold
                2: 1  # make amount positive
            }[actions['action']] * actions['amount']
            if not signal: signal = 0  # sometimes gives -0.0, dunno if that matters anywhere downstream
            # multi-action min_trade accounted for in constructor
        elif h.action_type == 'all_or_none':
            signal = {
                0: -step_acc.value,  # sell-all
                1: 0,  # hold
                2: step_acc.cash  # buy-all
            }[actions]

        step_acc.signals.append(float(signal))

        fee = {
            Exchange.GDAX: 0.0025,  # https://support.gdax.com/customer/en/portal/articles/2425097-what-are-the-fees-on-gdax-
            Exchange.KRAKEN: 0.0026  # https://www.kraken.com/en-us/help/fees
        }[EXCHANGE]
        reward = 0
        abs_sig = abs(signal)
        total_before = step_acc.cash + step_acc.value
        # Perform the trade. In training mode, we'll let it dip into negative here, but then kill and punish below.
        # In testing/live, we'll just block the trade if they can't afford it
        if signal > 0:
            if abs_sig <= step_acc.cash:
                step_acc.value += abs_sig - abs_sig*fee
                step_acc.cash -= abs_sig
            else:
                reward -= self.possible_reward
        elif signal < 0:
            if abs_sig <= step_acc.value:
                step_acc.cash += abs_sig - abs_sig*fee
                step_acc.value -= abs_sig
            else:
                reward -= self.possible_reward

        # teach it to not to do something it can't do (doesn't matter too much since we can just block the trade, but
        # hey - nicer if he "knows")
        if self.all_or_none and step_acc.last_action == actions and actions != 1:  # if buy->buy or sell->sell
            reward -= self.possible_reward

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        pct_change = self.prices_diff[step_acc.i + 1]

        step_acc.value += pct_change * step_acc.value
        total_now = step_acc.value + step_acc.cash
        step_acc.totals.trade.append(total_now)

        # calculate what the reward would be "if I held", to calculate the actual reward's _advantage_ over holding
        hold_before = step_acc.hold_value
        step_acc.hold_value += pct_change * hold_before
        step_acc.totals.hold.append(step_acc.hold_value + self.start_cash)

        # Reward is in dollar-change. As we build a great portfolio, the reward should get bigger and bigger (and
        # the agent should notice this)
        if h.reward_type == 'raw':
            reward += (total_now - total_before)
        elif h.reward_type == 'advantage':
            reward += (total_now - total_before) - (step_acc.hold_value - hold_before)
        elif h.reward_type == 'sharpe':
            # don't tally individual trade rewards for sharpe, it's calculated at the end and passed-back to all
            # steps via discount=1. We do want the other penalties above though
            pass

        step_acc.i += 1
        ep_acc.total_steps += 1

        stationary = [step_acc.last_action] if self.all_or_none else [step_acc.cash, step_acc.value]
        next_state = self.get_next_state(step_acc.i, stationary)
        if h.scale:
            reward = self.scaler.transform([reward], Scaler.REWARD)[0]

        # if h.reward_type == 'sharpe': reward = 0

        terminal = int(step_acc.i + 1 >= self.limit)
        if terminal and self.mode in (Mode.TRAIN, Mode.TEST):
            # We're done.
            step_acc.signals.append(0)  # Add one last signal (to match length)
            if h.reward_type == 'sharpe':
                diff = (pd.Series(step_acc.totals.trade).pct_change() - pd.Series(step_acc.totals.hold).pct_change())[1:]
                mean, std = diff.mean(), diff.std()
                if (std, mean) != (0, 0):
                    reward += mean / std


        if terminal and self.mode in (Mode.LIVE, Mode.TEST_LIVE):
            # Only do real buy/sell on last step if LIVE (in case there are multiple steps b/w, we only care about
            # present). Then we unset terminal, after we fetch some new data (keep going)
            # GDAX https://github.com/danpaquin/gdax-python
            live = self.mode == Mode.LIVE

            # Since we have a "ramp-up" window of data (to build the scaler & such), it'll make some fake trades
            # that don't go through. The first time we hit HEAD (an actual live timestep), we'll reset our numbers
            if not self.live_at_head:
                self.live_at_head = True
                print("Non-live advantage before reaching HEAD")
                self.episode_finished(None)
                step_acc.hold.cash = step_acc.cash = self.start_cash
                step_acc.hold.value = step_acc.value = self.start_value

            if signal < 0:
                print(f"Selling {signal}")
                if live:
                    res = self.gdax_client.sell(
                        # price=str(abs_sig),  # USD
                        type='market',
                        size=str(round(abs_sig, 4)),  # BTC .0
                        product_id='BTC-USD')
                    print(res)
            elif signal > 0:
                print(f"Buying {signal}")
                if live:
                    res = self.gdax_client.buy(
                        # price=str(abs_sig),  # USD
                        type='market',
                        size=str(round(abs_sig, 4)),  # BTC
                        product_id='BTC-USD')
                    print(res)
            elif ep_acc.total_steps % 10 == 0:
                print(".")

            if signal != 0 and live:
                self.update_btc_price()

            new_data = None
            while new_data is None:
                new_data, n_new, new_timestamp = data.fetch_more(
                    conn=self.conn, last_timestamp=self.last_timestamp, arbitrage=self.hypers.arbitrage)
                time.sleep(20)
            self.last_timestamp = new_timestamp
            self.df = pd.concat([self.df.iloc[-1000:], new_data], axis=0)  # shed some used data, add new
            self.observations, self.prices = self.xform_data(self.df)
            self.prices_diff = self.diff(self.prices, percent=True)
            step_acc.i = self.df.shape[0] - n_new - 1

            if live:
                accounts = self.gdax_client.get_accounts()
                step_acc.cash = float([a for a in accounts if a['currency'] == 'USD'][0]['balance']) / self.btc_price
                step_acc.value = float([a for a in accounts if a['currency'] == 'BTC'][0]['balance'])
            if signal != 0:
                print(f"New Total: {step_acc.cash + step_acc.value}")
                self.episode_finished(None)  # Fixme refactor, awkward function to call here
            next_state['stationary'] = [step_acc.cash, step_acc.value]
            terminal = False

        # if step_acc.value <= 0 or step_acc.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def episode_finished(self, runner):
        step_acc, ep_acc, test_acc = self.acc.step, self.acc.episode, self.acc.tests
        signals = step_acc.signals
        totals = step_acc.totals
        n_uniques = float(len(np.unique(signals)))

        # Calculate the Sharpe ratio.
        diff = (pd.Series(totals.trade).pct_change() - pd.Series(totals.hold).pct_change())[1:]
        mean, std, sharpe = diff.mean(), diff.std(), 0
        if (std, mean) != (0, 0):
            # Usually Sharpe has `sqrt(num_trades)` in front (or `num_trading_days`?). Experimenting being creative w/
            # trade-diversity, etc. Give Sharpe some extra info
            # breadth = math.sqrt(np.uniques(signals))
            # breadth = np.std([np.sign(x) for x in signals])  # get signal direction, amount not as important (and adds complications)
            breadth = 1
            sharpe = breadth * (mean / std)

        cumm_ret = (totals.trade[-1] / totals.trade[0] - 1) - (totals.hold[-1] / totals.hold[0] - 1)

        ep_acc.sharpes.append(float(sharpe))
        ep_acc.returns.append(float(cumm_ret))
        ep_acc.uniques.append(n_uniques)

        # Print (limit to note-worthy)
        lt_0 = len([s for s in signals if s < 0])
        eq_0 = len([s for s in signals if s == 0])
        gt_0 = len([s for s in signals if s > 0])
        completion = int(test_acc.i / test_acc.n_tests * 100)
        print(f"{completion}%\tSteps: {step_acc.i}\tSharpe: {'%.3f'%sharpe}\tReturn: {'%.3f'%cumm_ret}\tTrades:\t{lt_0}[<0]\t{eq_0}[=0]\t{gt_0}[>0]")
        return True

    def run_deterministic(self, runner, print_results=True):
        next_state, terminal = self.reset(), False
        while not terminal:
            next_state, terminal, reward = self.execute(runner.agent.act(next_state, deterministic=True))
        if print_results: self.episode_finished(None)

    def train_and_test(self, agent, n_steps, n_tests, early_stop):
        test_acc = self.acc.tests
        n_steps = n_steps * 10000
        test_acc.n_tests = n_tests
        test_acc.i = 0
        timesteps_each = n_steps // n_tests
        runner = Runner(agent=agent, environment=self)

        try:
            while test_acc.i <= n_tests:
                self.use_dataset(Mode.TRAIN)
                # max_episode_timesteps not required, since we kill on (cash|value)<0 or max_repeats
                runner.run(timesteps=timesteps_each)
                self.use_dataset(Mode.TEST)
                self.run_deterministic(runner, print_results=True)
                if early_stop > 0:
                    sharpes = np.array(self.acc.episode.sharpes[-early_stop:])
                    if test_acc.i >= early_stop and np.all(sharpes > 0):
                        test_acc.i = n_tests
                test_acc.i += 1
        except KeyboardInterrupt:
            # Lets us kill training with Ctrl-C and skip straight to the final test. This is useful in case you're
            # keeping an eye on terminal and see "there! right there, stop you found it!" (where early_stop & n_steps
            # are the more methodical approaches)
            pass

        # On last "how would it have done IRL?" run, without getting in the way (no killing on repeats, 0-balance)
        print('Running no-kill test-set')
        self.use_dataset(Mode.TEST, full_set=True)
        self.run_deterministic(runner, print_results=True)

    def run_live(self, agent, test=True):
        self.live_at_head = False
        self.acc.tests.n_tests = 1  # not used (but referenced for %completion)
        gdax_conf = data.config_json['GDAX']
        self.gdax_client = gdax.AuthenticatedClient(gdax_conf['key'], gdax_conf['b64secret'], gdax_conf['passphrase'])
        # self.gdax_client = gdax.AuthenticatedClient(gdax_conf['key'], gdax_conf['b64secret'], gdax_conf['passphrase'],
        #                                        api_url="https://api-public.sandbox.gdax.com")

        accounts = self.gdax_client.get_accounts()
        self.start_cash = float([a for a in accounts if a['currency'] == 'USD'][0]['balance']) / self.btc_price
        self.start_value = float([a for a in accounts if a['currency'] == 'BTC'][0]['balance'])
        print(f'Starting total: {self.start_cash + self.start_value}')

        runner = Runner(agent=agent, environment=self)
        self.use_dataset(Mode.TEST_LIVE if test else Mode.LIVE)
        self.run_deterministic(runner, print_results=True)
