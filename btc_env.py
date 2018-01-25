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

    # 400k should be enough data to safely say "I've seen it all, just scale (don't fit) going forward")
    STOP_AT = 3e5
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

# We don't want random-seeding for reproducability! We _want_ two runs to give different results, because we only
# trust the hyper combo which consistently gives positive results!
ALLOW_SEED = False
TIMESTEPS = int(2e6)


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
            step=dict(i=0)  # setup in reset()
        )
        self.mode = Mode.TRAIN
        self.conn = data.engine.connect()

        # TODO this might need to be placed somewhere that updates relatively often
        # gdax min order size = .01btc; krakken = .002btc
        self.min_trade = {Exchange.GDAX: .01, Exchange.KRAKEN: .002}[EXCHANGE]
        try:
            self.btc_price = int(requests.get(f"https://api.cryptowat.ch/markets/{EXCHANGE.value}/btcusd/price").json()['result']['price'])
        except:
            self.btc_price = 12000

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
                    ## Original indicators from some boilerplate repo I started with
                    self._diff(SMA(ind, timeperiod=15), percent),
                    self._diff(SMA(ind, timeperiod=60), percent),
                    self._diff(RSI(ind, timeperiod=14), percent),
                    self._diff(ATR(ind, timeperiod=14), percent),

                    ## Indicators from the book "How to Day Trade For a Living". Not sure which are more solid...
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

    def use_dataset(self, mode, no_kill=False):
        """Fetches, transforms, and stores the portion of data you'll be working with (ie, 80% train data, 20% test
        data, or the live database). Make sure to call this before reset()!
        """
        before_time = time.time()
        self.mode = mode
        self.no_kill = no_kill
        if mode in (Mode.LIVE, Mode.TEST_LIVE):
            self.conn = data.engine_live.connect()
            # Work with 6000 timesteps up until the present (play w/ diff numbers, depends on LSTM)
            # Offset=0 data.py currently pulls recent-to-oldest, then reverses
            limit, offset = (6000, 0) # if not self.conv2d else (self.hypers.step_window + 1, 0)
            df, self.last_timestamp = data.db_to_dataframe(
                self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage, last_timestamp=True)
            # save away for now so we can keep transforming it as we add new data (find a more efficient way)
            self.df = df
        else:
            self.row_ct = data.count_rows(self.conn, arbitrage=self.hypers.arbitrage)
            split = .9  # Using 90% training data.
            n_train, n_test = int(self.row_ct * split), int(self.row_ct * (1 - split))
            limit, offset = (n_test, n_train) if mode == mode.TEST else (n_train, 0)
            df = data.db_to_dataframe(self.conn, limit=limit, offset=offset, arbitrage=self.hypers.arbitrage)

        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._diff(self.prices, percent=True)
        after_time = round(time.time() - before_time)
        # print(f"Loading {mode.name} took {after_time}s")

    def reset(self):
        self.time = time.time()
        step_acc, ep_acc = self.acc.step, self.acc.episode
        # Cash & value are the real scores - how much we end up with at the end of an episode
        step_acc.cash, step_acc.value = self.start_cash, self.start_value
        # But for our purposes, we care more about "how much better is what we made than if we held". We're training
        # a trading bot, not an investing bot. So we compare these at the end, calling it "advantage"
        step_acc.hold = Box(value=self.start_cash, cash=self.start_value)
        # advance some steps just for cushion, various operations compare back a couple steps
        start_timestep = self.hypers.step_window if self.conv2d else 1
        step_acc.i = start_timestep
        step_acc.signals = [0] * start_timestep
        step_acc.repeats = 1
        ep_acc.i += 1

        first_state = self.observations[start_timestep]
        if self.hypers.scale:
            first_state = self.scaler.transform_state(first_state)
        if self.conv2d:
            # Take note of the +1 here. LSTM uses a single index [i], which grabs the list's end. Conv uses a window,
            # [-something:i], which _excludes_ the list's end (due to Python indexing). Without this +1, conv would
            # have a 1-step-behind delayed response.
            window = self.observations[start_timestep - self.hypers.step_window + 1:start_timestep + 1]
            first_state = np.expand_dims(window, axis=1)
        return dict(series=first_state, stationary=[1., 1., 0.])

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
            step_acc.repeats = 1  # reset repeat counter
        else:
            step_acc.repeats += 1

        step_acc.i += 1
        ep_acc.total_steps += 1
        # Is scaling here necessary, esp if using `hypers.scale`?
        cash_scaled, val_scaled = step_acc.cash / self.start_cash,  step_acc.value / self.start_value
        repeats_scaled = step_acc.repeats / self.hypers.punish_repeats

        next_state = self.observations[step_acc.i]
        if self.hypers.scale:
            next_state = self.scaler.transform_state(next_state)
            reward = self.scaler.transform_reward(reward)
        if self.conv2d:
            window = self.observations[step_acc.i - self.hypers.step_window + 1:step_acc.i + 1]
            next_state = np.expand_dims(window, axis=1)
        next_state = dict(series=next_state, stationary=[cash_scaled, val_scaled, repeats_scaled])

        terminal = int(step_acc.i + 1 >= len(self.observations))
        # Kill and punish if (a) agent ran out of money; (b) is doing nothing for way too long
        if not self.no_kill and (step_acc.cash < 0 or step_acc.value < 0 or step_acc.repeats >= self.hypers.punish_repeats):
            reward -= 1.  # BTC. Big punishment, like $12k
            terminal = True
        if terminal and self.mode in (Mode.TRAIN, Mode.TEST):
            # We're done.
            step_acc.signals.append(0)  # Add one last signal (to match length)
        if terminal and self.mode in (Mode.LIVE, Mode.TEST_LIVE):
            # Only do real buy/sell on last step if LIVE (in case there are multiple steps b/w, we only care about
            # present). Then we unset terminal, after we fetch some new data (keep going)
            # GDAX https://github.com/danpaquin/gdax-python
            live = self.mode == Mode.LIVE
            if signal < 0:
                if live:
                    self.gdax_client.sell(
                        # price=str(abs_sig),  # USD
                        size=float(abs_sig),  # BTC
                        product_id='BTC-USD')
                print(f"Sold {signal}!")
            elif signal > 0:
                if live:
                    self.gdax_client.buy(
                        # price=str(abs_sig),  # USD
                        size=float(abs_sig),  # BTC
                        product_id='BTC-USD')
                print(f"Bought {signal}!")
            elif step_acc.i % 10 == 0:
                print(".")

            new_data = None
            while new_data is None:
                new_data, n_new, new_timestamp = data.fetch_more(
                    conn=self.conn, last_timestamp=self.last_timestamp, arbitrage=self.hypers.arbitrage)
                time.sleep(20)
            self.last_timestamp = new_timestamp
            self.df = pd.concat([self.df, new_data], axis=0)
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
            next_state['stationary'] = [
                step_acc.cash / self.start_cash,
                step_acc.value / self.start_value,
                repeats_scaled  # TODO do I need to handle specifically for live?
            ]
            terminal = False

        # if step_acc.value <= 0 or step_acc.cash <= 0: terminal = 1
        return next_state, terminal, reward

    def episode_finished(self, runner):
        step_acc, ep_acc = self.acc.step, self.acc.episode
        time_ = round(time.time() - self.time)
        signals = step_acc.signals

        advantage = ((step_acc.cash + step_acc.value) - (self.start_cash + self.start_value)) - \
                    ((step_acc.hold.value + step_acc.hold.cash) - (self.start_cash + self.start_value))
        self.acc.episode.advantages.append(advantage)
        n_uniques = float(len(np.unique(signals)))
        self.acc.episode.uniques.append(n_uniques)

        # Print (limit to note-worthy)
        common = dict((round(k,2), v) for k, v in Counter(signals).most_common(5))
        completion = f"|{int(ep_acc.total_steps / TIMESTEPS * 100)}%"
        print(f"{ep_acc.i}|⌛:{step_acc.i}{completion}\tA:{'%.3f'%advantage}\t{common}({n_uniques}uniq)")
        return True

    def run_deterministic(self, runner, print_results=True):
        next_state, terminal = self.reset(), False
        while not terminal:
            next_state, terminal, reward = self.execute(runner.agent.act(next_state, deterministic=True))
        if print_results: self.episode_finished(None)

    def train_and_test(self, agent, early_stop=-1, n_tests=40):
        n_train = TIMESTEPS // n_tests
        i = 0
        runner = Runner(agent=agent, environment=self)

        try:
            while i <= n_tests:
                self.use_dataset(Mode.TRAIN)
                runner.run(timesteps=n_train, max_episode_timesteps=n_train)
                self.use_dataset(Mode.TEST)
                self.run_deterministic(runner, print_results=True)
                if early_stop > 0:
                    advantages = np.array(self.acc.episode.advantages[-early_stop:])
                    if i >= early_stop and np.all(advantages > 0):
                        i = n_tests
                i += 1
        except KeyboardInterrupt:
            # Lets us kill training with Ctrl-C and skip straight to the final test. This is useful in case you're
            # keeping an eye on terminal and see "there! right there, stop you found it!" (where early_stop & n_tests
            # are the more methodical approaches)
            pass

        # On last "how would it have done IRL?" run, without getting in the way (no killing on repeats, 0-balance)
        print('Running no-kill test-set')
        self.use_dataset(Mode.TEST, no_kill=True)
        self.run_deterministic(runner, print_results=True)

    def run_live(self, agent, test=True):
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
