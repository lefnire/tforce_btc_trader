"""BTC trading environment. Trains on BTC price history to learn to buy/sell/hold.

This is an environment tailored towards TensorForce, not OpenAI Gym. Gym environments are
a standard used by many projects (Baselines, Coach, etc) and so would make sense to use; and TForce is compatible with
Gym envs. It's just that there's hoops to go through converting a Gym env to TForce, and it was ugly code. I actually
had it that way, you can search through Git if you want the Gym env; but one day I decided "I'm not having success with
any of these other projects, TForce is the best - I'm just gonna stick to that" and this approach was cleaner.

I actually do want to try NervanaSystems/Coach, that one's new since I started developing. Will require converting this
env back to Gym format. Anyone wanna give it a go?
"""

import random, time, requests, pdb, gdax, math, pickle, os, shutil
from scipy.stats import truncnorm
from enum import Enum
import numpy as np
import pandas as pd
import talib.abstract as tlib
from box import Box
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
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
    """

    # At some point we can safely say "I've seen enough, just scale (don't fit) going forward")
    STOP_AT = int(5e5)

    # state types
    class Type(Enum):
        REWARD = 1
        STATIONARY_1 = 2
        STATIONARY_2 = 3
        FINAL_REWARD = 4

    def __init__(self, recreate=False):
        self.scalers = {}
        self.file_data = {}
        self.buffer_data = {}
        self.i = {}

        # (re)create folder
        if recreate:
            try: shutil.rmtree(self.dirpath())
            except: pass
        os.makedirs(self.dirpath(), exist_ok=True)

        for k in Scaler.Type:
            # RobustScaler will clean up outliers (not too much; we want to keep big winners, just discard corruptions)
            # MinMax will bring it to (-1,1) so we can weight reward-types relatively.
            #self.scalers[k] = make_pipeline(
            #    preprocessing.RobustScaler(quantile_range=(1.,99.)),
            #    preprocessing.MinMaxScaler(feature_range=(-1,1))
            #)
            self.scalers[k] = preprocessing.QuantileTransformer()
            #self.scalers[k] = preprocessing.MinMaxScaler(feature_range=(-1,1))
            self.buffer_data[k] = []
            self.i[k] = 0

            # Load prior runs' data to seed from
            filepath = self.filepath(k)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.file_data[k] = pickle.load(f)
            else:
                self.file_data[k] = []

    def dirpath(self):
        return os.path.join(os.getcwd(), "saves", "scalers")

    def filepath(self, k):
        return os.path.join(self.dirpath(), f'{k.name}.pkl')

    @staticmethod
    def transform_series(states):
        """Transform the price-series stuff separate from the other stuff (rewards, stationary, etc). (1) Series needs
        to happen just once on init; (2) series needs special outlier-handling (price diffs go [-inf,inf] etc; just
        remove them with RobustScaler. The other things are unlikely to have crazy outliers
        """
        return preprocessing.robust_scale(states, quantile_range=(3., 97.))

    def is_done(self, k):
        """After we've fitted enough (see STOP_AT), start returning direct-transforms for performance improvement"""
        if self.file_data[k] == -1:
            return True
        if len(self.file_data[k]) > self.STOP_AT:
            self.scalers[k].fit(self.file_data[k])
            self.file_data[k] = self.buffer_data[k] = -1
            return True
        return False

    def transform(self, input, k, skip_every=50, save_every=20000):
        scaler = self.scalers[k]
        if self.is_done(k):
            return scaler.transform([input])[0]
        self.buffer_data[k].append(input)
        self.i[k] += 1

        # Skip every x fittings. Each individual doesn't contribute a whole lot anyway, and costs a lot
        if self.i[k] > skip_every and self.i[k] % skip_every != 0:
            return scaler.transform([input])[0]

        # Save the scaler every so often
        if self.i[k] % save_every == 0:
            filepath = self.filepath(k)
            file_data = self.file_data[k]
            if os.path.exists(filepath):
                try:
                    # FIXME this keeps crashing w/ "EOFError: Ran out of input"; investigate
                    with open(filepath, 'rb') as f:
                        file_data = pickle.load(f)
                except:
                    return scaler.transform([input])[0]
            self.file_data[k] = file_data + self.buffer_data[k]
            with open(filepath, 'wb') as f:
                pickle.dump(self.file_data[k], f)
            self.buffer_data[k] = []

        scaler.fit(self.file_data[k] + self.buffer_data[k])
        return scaler.transform([input])[0]


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
                uniques=[],
                custom_scores=[]
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

        self.scaler = Scaler(recreate=cli_args.clear_scalers)

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
        states = Scaler.transform_series(states)

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

        type_ = Scaler.Type.STATIONARY_1 if len(stationary) == 1 else Scaler.Type.STATIONARY_2
        stationary = self.scaler.transform(stationary, type_).tolist()

        if self.conv2d:
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
            # don't add per-trade rewards for sharpe, it's calculated at the end and passed-back to all
            # steps via discount=1. We do want the penalties above though
            pass

        step_acc.i += 1
        ep_acc.total_steps += 1

        stationary = [step_acc.last_action] if self.all_or_none else [step_acc.cash, step_acc.value]
        next_state = self.get_next_state(step_acc.i, stationary)

        # Scale
        if h.reward_type == 'sharpe':
            # since the reward comes at the end, scaling these rewards (which are just the penalties) brings
            # them to [-1,1]; on-par with the actual sharpe reward. Ensure they're small relative penalties.
            reward = -.001 if reward < 0. else 0.
        else:
            reward = self.scaler.transform([reward], Scaler.Type.REWARD)[0]

        terminal = int(step_acc.i + 1 >= self.limit)
        if terminal and self.mode in (Mode.TRAIN, Mode.TEST):
            # We're done.
            step_acc.signals.append(0)  # Add one last signal (to match length)
            custom = self.end_episode_score()
            ep_acc.custom_scores.append(float(custom))
            if h.reward_type == 'sharpe':
                reward += custom

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

    def sharpe(self):
        totals = self.acc.step.totals
        diff = (pd.Series(totals.trade).pct_change() - pd.Series(totals.hold).pct_change())[1:]
        mean, std = diff.mean(), diff.std()
        if (std, mean) != (0, 0):
            # Usually Sharpe has `sqrt(num_trades)` in front (or `num_trading_days`?). Experimenting being creative w/
            # trade-diversity, etc. Give Sharpe some extra info
            # breadth = math.sqrt(np.uniques(signals))  # standard
            breadth = 1  # disable
            return breadth * (mean / std)
        return 0.

    def end_episode_score(self):
        # These modifications help coax the agent in the right direction; kinda steering him with bumpers. "We
        # want you to hold more often than not, but more trades is better, and of course make money, ..."
        sharpe = self.sharpe()
        signs = [np.sign(x) for x in self.acc.step.signals]  # signal directions. amounts add complication
        mean_near_zero = 1 - abs(np.mean(signs))  # favor a mean closer to zero (more holds than not)
        diversity = np.sqrt(np.std(signs))  # favor more trades - up to a point (sqrt)

        # Scale them
        sharpe, mean_near_zero, diversity = self.scaler.transform(
            [sharpe, mean_near_zero, diversity], Scaler.Type.FINAL_REWARD, skip_every=1, save_every=5)
        # now they're scaled, could weight them by relative importance

        score = sharpe + mean_near_zero  # + diversity * .25
        if sharpe > 0.: score += .5
        return score

    def episode_finished(self, runner):
        step_acc, ep_acc, test_acc = self.acc.step, self.acc.episode, self.acc.tests
        signals = step_acc.signals
        totals = step_acc.totals
        n_uniques = float(len(np.unique(signals)))
        sharpe = self.sharpe()
        custom = ep_acc.custom_scores[-1]
        cumm_ret = (totals.trade[-1] / totals.trade[0] - 1) - (totals.hold[-1] / totals.hold[0] - 1)

        ep_acc.sharpes.append(float(sharpe))
        ep_acc.returns.append(float(cumm_ret))
        ep_acc.uniques.append(n_uniques)

        # Print (limit to note-worthy)
        lt_0 = len([s for s in signals if s < 0])
        eq_0 = len([s for s in signals if s == 0])
        gt_0 = len([s for s in signals if s > 0])
        completion = int(test_acc.i / test_acc.n_tests * 100)
        steps = ""  # f"\tSteps: {step_acc.i}"
        print(f"{completion}%{steps}\tCustom: {'%.3f'%custom}\tSharpe: {'%.3f'%sharpe}\tReturn: {'%.3f'%cumm_ret}\tTrades:\t{lt_0}[<0]\t{eq_0}[=0]\t{gt_0}[>0]")
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
