"""BTC trading environment. Trains on BTC price history to learn to buy/sell/hold.

This is an environment tailored towards TensorForce, not OpenAI Gym. Gym environments are
a standard used by many projects (Baselines, Coach, etc) and so would make sense to use; and TForce is compatible with
Gym envs. It's just that there's hoops to go through converting a Gym env to TForce, and it was ugly code. I actually
had it that way, you can search through Git if you want the Gym env; but one day I decided "I'm not having success with
any of these other projects, TForce is the best - I'm just gonna stick to that" and this approach was cleaner.

I actually do want to try NervanaSystems/Coach, that one's new since I started developing. Will require converting this
env back to Gym format. Anyone wanna give it a go?
"""

import random, time, requests, pdb, gdax
from enum import Enum
import numpy as np
import pandas as pd
from talib.abstract import SMA, RSI, ATR, EMA
from collections import Counter
import tensorflow as tf
from box import Box
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from sklearn.preprocessing import RobustScaler, robust_scale
from data.data import Exchange, EXCHANGE
from data import data


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
            self.SERIES: RobustScaler(quantile_range=(5., 95.)),
            self.STATIONARY: RobustScaler(quantile_range=(5., 95.))
        }
        self.data = {
            self.REWARD: [],
            self.SERIES: [],
            self.STATIONARY: []
        }
        self.done = False
        self.i = 0

    def _should_skip(self):
        # After we've fitted enough (see STOP_AT), start returning direct-transforms for performance improvement
        # Skip every few fittings. Each individual doesn't contribute a whole lot anyway, and costs a lot
        return self.done or (self.i % self.SKIP != 0 and self.i > self.SKIP)

    def transform(self, input, kind):
        # this is awkward; we only want to increment once per step, but we're calling this fn 3x per step (once
        # for series, once for stationary, once for reward). Explicitly saying "only increment for one of those" here.
        # Using STATIONARY since SERIES might be called once per timestep in a conv window. TODO Seriously awkward
        if kind == self.STATIONARY: self.i += 1

        scaler = self.scalers[kind]
        matrix = np.array(input).ndim == 2
        if self._should_skip():
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

# We don't want random-seeding for reproducibilityy! We _want_ two runs to give different results, because we only
# trust the hyper combo which consistently gives positive results!
ALLOW_SEED = False


class BitcoinEnv(Environment):
    def __init__(self, hypers, name='ppo_agent'):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.hypers = Box(hypers)
        self.conv2d = self.hypers['net.type'] == 'conv2d'
        self.agent_name = name

        # cash/val start @ about $3.5k each. You should increase/decrease depending on how much you'll put into your
        # exchange accounts to trade with. Presumably the agent will learn to work with what you've got (cash/value
        # are state inputs); but starting capital does effect the learning process.
        self.start_cash, self.start_value = .3, .3

        # We have these "accumulator" objects, which collect values over steps, over episodes, etc. Easier to keep
        # same-named variables separate this way.
        self.acc = Box(
            episode=dict(
                i=0,
                total_steps=0,
                advantages=[],
                uniques=[]
            ),
            step=dict(i=0),  # setup in reset()
            tests=dict(
                i=0,
                n_tests=0
            )
        )
        self.mode = Mode.TRAIN
        self.conn = data.engine.connect()

        # gdax min order size = .01btc; krakken = .002btc
        self.min_trade = {Exchange.GDAX: .01, Exchange.KRAKEN: .002}[EXCHANGE]
        self.update_btc_price()

        # Action space
        trade_cap = self.min_trade * 2  # not necessary to limit it like this, doing for my own sanity in live-mode
        if self.hypers.single_action:
            # In single_action we discard any vals b/w [-min_trade, +min_trade] and call it "hold" (in execute())
            self.actions_ = dict(type='float', shape=(), min_value=-trade_cap, max_value=trade_cap)
        else:
            # In multi-modal, hold is an actual action (in which case we discard "amount")
            self.actions_ = dict(
                action=dict(type='int', shape=(), num_actions=3),
                amount=dict(type='float', shape=(), min_value=self.min_trade, max_value=trade_cap))

        # Observation space
        self.cols_ = data.n_cols(indicators=self.hypers.indicators, arbitrage=self.hypers.arbitrage)
        self.states_ = dict(
            series=dict(type='float', shape=self.cols_),  # all state values that are time-ish
            stationary=dict(type='float', shape=3)  # everything that doesn't care about time (cash, value, n_repeats)
        )

        if self.conv2d:
            # width = step-window (150 time-steps)
            # height = nothing (1)
            # channels = features/inputs (price actions, OHCLV, etc).
            self.states_['series']['shape'] = (self.hypers.step_window, 1, self.cols_)
            if self.hypers.repeat_last_state:
                self.states_['stationary']['shape'] += self.cols_

        # Should be one scaler for any permutation of data (since the columns need to align exactly)
        scaler_k = f'{self.hypers.arbitrage}|{self.hypers.indicators}|{self.hypers.repeat_last_state}'
        if scaler_k not in scalers:
            scalers[scaler_k] = Scaler()
        self.scaler = scalers[scaler_k]

        # Calculate a possible reward to be used as an average for repeat-punishing
        prices = data.db_to_dataframe(self.conn, arbitrage=self.hypers.arbitrage)[data.target].values
        prices_diff = self._diff(prices, percent=True)
        self.possible_reward = self.start_value * np.median([p for p in prices_diff if p > 0])
        print('possible_reward', self.possible_reward)

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

    def update_btc_price(self):
        try:
            self.btc_price = int(requests.get(f"https://api.cryptowat.ch/markets/{EXCHANGE.value}/btcusd/price").json()['result']['price'])
        except:
            self.btc_price = self.btc_price or 8000

    def _diff(self, arr, percent=False):
        series = pd.Series(arr)
        diff = series.pct_change() if percent else series.diff()
        diff.iloc[0] = 0  # always NaN, nothing to compare to

        # Remove outliers (turn them to NaN)
        q = diff.quantile(0.99)
        diff = diff.mask(diff > q, np.nan)

        # then forward-fill the NaNs.
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
                    # TODO this is my naive approach, I'm not a TA expert. Could use a second pair of eyes
                    self._diff(SMA(ind, timeperiod=self.hypers.indicators), percent),
                    self._diff(EMA(ind, timeperiod=self.hypers.indicators), percent),
                    self._diff(RSI(ind, timeperiod=self.hypers.indicators), percent),
                    self._diff(ATR(ind, timeperiod=self.hypers.indicators), percent),
                ]

        states = np.nan_to_num(np.column_stack(columns))
        prices = df[data.target].values

        # Pre-scale all price actions up-front, since they don't change. We'll scale changing values real-time elsewhere
        if self.hypers.scale:
            states = self.scaler.transform(states, Scaler.SERIES)

        return states, prices

    def use_dataset(self, mode, no_kill=False):
        """Fetches, transforms, and stores the portion of data you'll be working with (ie, 80% train data, 20% test
        data, or the live database). Make sure to call this before reset()!
        """
        self.mode = mode
        self.no_kill = no_kill
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
                limit, offset = n_test, n_train
                if no_kill is False:
                    limit = 50000  # he's not likely to get past that, so save some RAM (=time)
            else:
                # Grab a random window from the 90% training data. The random bit is important so the agent
                # sees a variety of data. The window-size bit is a hack: as long as the agent doesn't die (doesn't cause
                # `terminal=True`), PPO's MemoryModel can keep filling up until it crashes TensorFlow. This ensures
                # there's a stopping point (limit). I'd rather see how far he can get w/o dying, figure out a solution.
                limit = 25000
                offset = random.randint(0, n_train - limit)
            df = data.db_to_dataframe(self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage)

        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._diff(self.prices, percent=True)

    def _get_next_state(self, i, cash, value, repeats):
        series = self.observations[i]
        stationary = [cash, value, repeats]
        if self.hypers.scale:
            # series already scaled in self._xform_data()
            stationary = self.scaler.transform(stationary, Scaler.STATIONARY).tolist()

        if self.conv2d:
            if self.hypers.repeat_last_state:
                stationary += series.tolist()
            # Take note of the +1 here. LSTM uses a single index [i], which grabs the list's end. Conv uses a window,
            # [-something:i], which _excludes_ the list's end (due to Python indexing). Without this +1, conv would
            # have a 1-step-behind delayed response.
            window = self.observations[i - self.hypers.step_window + 1:i + 1]
            series = np.expand_dims(window, axis=1)
        return dict(series=series, stationary=stationary)

    def reset(self):
        step_acc, ep_acc = self.acc.step, self.acc.episode
        # Cash & value are the real scores - how much we end up with at the end of an episode
        step_acc.cash, step_acc.value = self.start_cash, self.start_value
        # But for our purposes, we care more about "how much better is what we made than if we held". We're training
        # a trading bot, not an investing bot. So we compare these at the end, calling it "advantage"
        step_acc.hold = Box(value=self.start_cash, cash=self.start_value)
        start_timestep = 1
        if self.conv2d:
            # for conv2d, start at the end of the first window (grab a full window)
            start_timestep = self.hypers.step_window
        if self.hypers.indicators:
            # if using indicators, add said window as padding so our first timestep has indicator data
            start_timestep += int(self.hypers.indicators)
        step_acc.i = start_timestep
        step_acc.signals = [0] * start_timestep
        step_acc.repeats = 0
        ep_acc.i += 1

        return self._get_next_state(start_timestep, self.start_cash, self.start_value, 0.)

    def execute(self, actions):
        if self.hypers.single_action:
            signal = 0 if -self.min_trade < actions < self.min_trade else actions
        else:
            # Two actions: `action` (buy/sell/hold) and `amount` (how much)
            signal = {
                0: -1,  # make amount negative
                1: 0,  # hold
                2: 1  # make amount positive
            }[actions['action']] * actions['amount']
            if not signal: signal = 0  # sometimes gives -0.0, dunno if that matters anywhere downstream
            # multi-action min_trade accounted for in constructor

        step_acc, ep_acc = self.acc.step, self.acc.episode

        step_acc.signals.append(float(signal))

        fee = {
            Exchange.GDAX: 0.0025,  # https://support.gdax.com/customer/en/portal/articles/2425097-what-are-the-fees-on-gdax-
            Exchange.KRAKEN: 0.0026  # https://www.kraken.com/en-us/help/fees
        }[EXCHANGE]
        reward = 0
        abs_sig = abs(signal)
        before = Box(cash=step_acc.cash, value=step_acc.value, total=step_acc.cash+step_acc.value)
        # Perform the trade. In training mode, we'll let it dip into negative here, but then kill and punish below.
        # In testing/live, we'll just block the trade if they can't afford it
        if signal > 0 and not (self.no_kill and abs_sig > step_acc.cash):
            step_acc.value += abs_sig - abs_sig*fee
            step_acc.cash -= abs_sig
        elif signal < 0 and not (self.no_kill and abs_sig > step_acc.value):
            step_acc.cash += abs_sig - abs_sig*fee
            step_acc.value -= abs_sig

        # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        diff_loc = step_acc.i + 1
        pct_change = self.prices_diff[diff_loc]
        step_acc.value += pct_change * step_acc.value
        total = step_acc.value + step_acc.cash
        reward += total - before.total

        # calculate what the reward would be "if I held", to calculate the actual reward's _advantage_ over holding
        before = step_acc.hold
        before.value += pct_change * before.value

        # Collect repeated same-action count (homogeneous actions punished below)
        recent_actions = np.array(step_acc.signals[-step_acc.repeats:])
        if np.any(recent_actions > 0) and np.any(recent_actions < 0) and np.any(recent_actions == 0):
            step_acc.repeats = 0  # reset repeat counter
        else:
            step_acc.repeats += 1
            # by the time we hit punish_repeats, we're doubling punishments / canceling rewards. Note: we don't want to
            # multiply by `reward` here because repeats are often 0, which means 0 penalty. Hence `possible_reward`
            repeat_penalty = self.possible_reward * (step_acc.repeats / self.hypers.punish_repeats)
            reward -= repeat_penalty
            # step_acc.value -= repeat_penalty  # TMP: experimenting w/ showing the human & BO

        step_acc.i += 1
        ep_acc.total_steps += 1

        next_state = self._get_next_state(step_acc.i, step_acc.cash, step_acc.value, step_acc.repeats)
        if self.hypers.scale:
            reward = self.scaler.transform([reward], Scaler.REWARD)[0]

        terminal = int(step_acc.i + 1 >= len(self.observations))
        # Kill and punish if (a) agent ran out of money; (b) is doing nothing for way too long
        # The repeats bit isn't just for punishment, but because training can get stuck too long on losers
        if not self.no_kill and (step_acc.cash < 0 or step_acc.value < 0 or step_acc.repeats >= self.hypers.punish_repeats):
            reward -= 1.  # Big penalty. BTC, like $12k
            terminal = True
        if terminal and self.mode in (Mode.TRAIN, Mode.TEST):
            # We're done.
            step_acc.signals.append(0)  # Add one last signal (to match length)
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
            self.observations, self.prices = self._xform_data(self.df)
            self.prices_diff = self._diff(self.prices, percent=True)
            step_acc.i = self.df.shape[0] - n_new - 1

            if live:
                accounts = self.gdax_client.get_accounts()
                step_acc.cash = float([a for a in accounts if a['currency'] == 'USD'][0]['balance']) / self.btc_price
                step_acc.value = float([a for a in accounts if a['currency'] == 'BTC'][0]['balance'])
            if signal != 0:
                print(f"New Total: {step_acc.cash + step_acc.value}")
                self.episode_finished(None)  # Fixme refactor, awkward function to call here
            next_state['stationary'] = [step_acc.cash, step_acc.value, step_acc.repeats]
            terminal = False

        # if step_acc.value <= 0 or step_acc.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def episode_finished(self, runner):
        step_acc, ep_acc, test_acc = self.acc.step, self.acc.episode, self.acc.tests
        signals = step_acc.signals

        advantage = ((step_acc.cash + step_acc.value) - (self.start_cash + self.start_value)) - \
                    ((step_acc.hold.value + step_acc.hold.cash) - (self.start_cash + self.start_value))
        # per step average advantage, then bring it to a reasonable number (up from ~.0001)
        advantage = advantage / step_acc.i * 10000
        if advantage == 0.: advantage = -.01  # no HODLing!
        self.acc.episode.advantages.append(advantage)
        n_uniques = float(len(np.unique(signals)))
        self.acc.episode.uniques.append(n_uniques)

        # Print (limit to note-worthy)
        lt_0 = len([s for s in signals if s < 0])
        eq_0 = len([s for s in signals if s == 0])
        gt_0 = len([s for s in signals if s > 0])
        completion = int(test_acc.i / test_acc.n_tests * 100)
        print(f"{completion}%\tSteps: {step_acc.i}\tAdvantage: {'%.3f'%advantage}\tTrades:\t{lt_0}[<0]\t{eq_0}[=0]\t{gt_0}[>0]")
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
                    advantages = np.array(self.acc.episode.advantages[-early_stop:])
                    if test_acc.i >= early_stop and np.all(advantages > 0):
                        test_acc.i = n_tests
                test_acc.i += 1
        except KeyboardInterrupt:
            # Lets us kill training with Ctrl-C and skip straight to the final test. This is useful in case you're
            # keeping an eye on terminal and see "there! right there, stop you found it!" (where early_stop & n_steps
            # are the more methodical approaches)
            pass

        # On last "how would it have done IRL?" run, without getting in the way (no killing on repeats, 0-balance)
        print('Running no-kill test-set')
        self.use_dataset(Mode.TEST, no_kill=True)
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
        self.use_dataset(Mode.TEST_LIVE if test else Mode.LIVE, no_kill=True)
        self.run_deterministic(runner, print_results=True)
