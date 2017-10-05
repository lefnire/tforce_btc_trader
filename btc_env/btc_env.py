import random, time, gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talib.abstract import SMA, RSI, ATR, EMA
from sklearn import preprocessing
from sklearn.externals import joblib
from collections import Counter
from sqlalchemy.sql import text
import tensorflow as tf
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce import util, TensorForceError
from tensorforce.environments import Environment

import data
from data import conn

try:
    scaler = joblib.load('data_/scaler.pkl')
except Exception: pass

try:
    min_max = joblib.load('data_/min_max.pkl')
    min_max_scaled = joblib.load('data_/min_max_scaled.pkl')
except Exception:
    min_max = None

class BitcoinEnv(gym.Env):
    metadata = {
        'render.modes': []
    }

    # Calling gym.make(ID) doesn't allow passing in params, so we make then initialize separately
    # def __init__(self):

    def init(self, gym_env, steps=2048*5+5, agent_name='PPOAgent|main', scale_features=False,
             indicators=False, start_cap=1e3, is_main=True, log_results=True, log_states=False):
        """Initialize hyperparameters (done here instead of __init__ since OpenAI-Gym controls instantiation)"""
        self.gym_env = gym_env
        self.steps = steps
        self.agent_name = agent_name
        self.scale_features = scale_features
        self.indicators = indicators
        self.start_cap = start_cap
        self.is_main = is_main
        self.log_results = log_results
        self.summary_writer = tf.summary.FileWriter(f"./a3c/saves/train/{agent_name}")
        self.log_states = log_states
        self.episode_results = {'cash': [], 'values': [], 'rewards': []}

        if 'DQN' in agent_name:
            gym_env.action_space = spaces.Discrete(4)
        else:
            gym_env.action_space = spaces.Box(low=-100, high=100, shape=(1,))
        gym_env.observation_space = spaces.Box(*min_max_scaled) if scale_features else\
            spaces.Box(*min_max) if min_max else\
            spaces.Box(low=-100, high=100, shape=(self.num_features(),))
        if scale_features: print('using min_max', min_max_scaled)
        elif min_max: print('using min_max', min_max)
        
        # self._seed()
        if is_main:
            data.wipe_rows(agent_name)

    def __str__(self): return 'BitcoinEnv'
    def _close(self): pass
    def _render(self, mode='human', close=False): pass
    def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def num_features(self):
        num = len(data.columns)
        if self.indicators:
            num += 4  # num features from self._get_indicators
        num *= len(data.tables)  # That many features per table
        num += 2  # [self.cash, self.value]
        return num

    @staticmethod
    def _pct_change(arr):
        return pd.Series(arr).pct_change()\
            .replace([np.inf, -np.inf, np.nan], [1., -1., 0.]).values

    @staticmethod
    def _diff(arr):
        return pd.Series(arr).diff()\
            .replace([np.inf, -np.inf], np.nan).ffill()\
            .fillna(0).values

    def _xform_data(self, df):
        columns = []
        for k in data.tables:
            # TA-Lib requires specifically-named columns (OHLCV)
            c = dict([(f'{k}_{c}', c) for c in data.columns if c != data.close_col])
            c[f'{k}_{data.close_col}'] = 'close'
            xchange_df = df.rename(columns=c)

            # Currently NO indicators works better (LSTM learns the indicators itself). I'm thinking because indicators
            # are absolute values, causing number-range instability
            columns += list(map(lambda k: self._diff(xchange_df[k]), c.values()))
            if self.indicators:
                columns += self._get_indicators(xchange_df)

        states = np.nan_to_num(np.column_stack(columns))
        prices = df[data.predict_col].values
        # Note: don't scale/normalize here, since we'll normalize w/ self.price/self.cash after each action
        return states, prices

    def _get_indicators(self, df):
        return [
            ## Original indicators from boilerplate
            self._diff(SMA(df, timeperiod=15)),
            self._diff(SMA(df, timeperiod=60)),
            self._diff(RSI(df, timeperiod=14)),
            self._diff(ATR(df, timeperiod=14)),

            ## Indicators from "How to Day Trade For a Living" (try these)
            ## Price, Volume, 9-EMA, 20-EMA, 50-SMA, 200-SMA, VWAP, prior-day-close
            # self._diff(EMA(df, timeperiod=9)),
            # self._diff(EMA(df, timeperiod=20)),
            # self._diff(SMA(df, timeperiod=50)),
            # self._diff(SMA(df, timeperiod=200)),
        ]

    def _reset(self):
        self.time = time.time()
        self.cash = self.value = self.start_cap
        self.cash_true = self.value_true = self.start_cap
        start_timestep = 1  # advance some steps just for cushion, various operations compare back a couple steps
        self.timestep = start_timestep
        self.signals = [0] * start_timestep
        self.total_reward = self.total_reward_true = 0

        # Fetch random slice of rows from the database (based on limit)
        offset = random.randint(0, data.count_rows() - self.steps)
        df = data.db_to_dataframe(limit=self.steps, offset=offset)
        self.observations, self.prices = self._xform_data(df)
        self.prices_diff = self._pct_change(self.prices)

        first_state = np.append(self.observations[start_timestep], [0., 0.])
        if self.scale_features:
            first_state = scaler.transform([first_state])[0]
        return first_state

    def _step(self, action):
        if type(self.gym_env.action_space) == spaces.Discrete:
            signal = {0: -40, 1: 0, 2: 5, 3: 40}[int(action)]
            signal_true = signal
        else:
            # signal = 0 if -40 < action < 1 else action
            signal = action[0]
            signal_true = action[1] if len(action) > 1 else signal
        self.signals.append(signal_true)

        fee = 0.0025  # https://www.gdax.com/fees/BTC-USD
        abs_sig = abs(signal)
        before = dict(cash=self.cash, value=self.value, total=self.cash+self.value)
        if signal > 0:
            if self.cash >= abs_sig:
                self.value += abs_sig - abs_sig*fee
            self.cash -= abs_sig
        elif signal < 0:
            if self.value >= abs_sig:
                self.cash += abs_sig - abs_sig*fee
            self.value -= abs_sig

        abs_sig_true = abs(signal_true)
        before_true = dict(cash=self.cash_true, value=self.value_true, total=self.cash_true + self.value_true)
        if signal_true > 0:
            if self.cash_true >= abs_sig_true:
                self.value_true += abs_sig_true - abs_sig_true * fee
            self.cash_true -= abs_sig_true
        elif signal_true < 0:
            if self.value_true >= abs_sig_true:
                self.cash_true += abs_sig_true - abs_sig_true * fee
            self.value_true -= abs_sig_true

        pct_change = self.prices_diff[self.timestep + 1]  # next delta. [1,2,2].pct_change() == [NaN, 1, 0]
        self.value += pct_change * self.value
        total = self.value + self.cash
        reward = total - before['total']  # Relative reward (seems to work better)

        self.value_true += pct_change * self.value_true
        total_true = self.value_true + self.cash_true
        reward_true = total_true - before_true['total']  # Relative reward (seems to work better)

        self.timestep += 1
        next_state = np.append(self.observations[self.timestep], [
            self.cash,  # 0 if before['cash'] == 0 else (self.cash - before['cash']) / before['cash'],
            self.value  # 0 if before['value'] == 0 else (self.value - before['value']) / before['value'],
        ])

        # If we need to record a few thousand observations for use in scaling or determining min/max vals (turn off after)
        if self.log_states:
            # create table if not exists observations (obs double precision[])
            obs = [float(o) for o in next_state]
            conn.execute(text("insert into observations (obs) values (:obs)"), obs=obs)

        if self.scale_features:
            next_state = scaler.transform([next_state])[0]

        self.total_reward += reward
        self.total_reward_true += reward_true

        terminal = int(self.timestep + 1 >= len(self.observations))
        if terminal:
            self.signals.append(0)  # Add one last signal (to match length)
            self.episode_results['cash'].append(self.cash_true)
            self.episode_results['values'].append(self.value_true)
            self.episode_results['rewards'].append(self.total_reward_true)
            self.time = round(time.time() - self.time)
            self._write_results()
        # if self.value <= 0 or self.cash <= 0: terminal = 1
        return next_state, reward, terminal, {}


    def _write_results(self):
        res = self.episode_results
        episode = len(res['cash'])
        # skip some for performance
        # if len(res['cash']) % 10 != 0: return

        reward, cash, value = float(self.total_reward), float(self.cash), float(self.value)
        total = cash + value
        avg100 = int(np.mean(res['cash'][-100:]) + np.mean(res['values'][-100:]))
        common = dict((round(k), v) for k, v in Counter(self.signals).most_common(5))
        high, low = np.max(self.signals), np.min(self.signals)
        # if np.isnan(high) or np.isnan(low): import pdb;pdb.set_trace()
        print(f"{episode}\t⌛:{self.time}s\tR:{int(reward)}\t${int(total)}\tAVG$:{avg100}\tActions:{common}(high={high},low={low})")

        if self.is_main:
            if self.log_results:
                results = self.episode_results
                total = float(results['cash'][-1] + results['values'][-1])
                reward = float(results['rewards'][-1])
                reward_avg = np.mean(results['rewards'][-20:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Total', simple_value=total)
                summary.value.add(tag='Perf/Reward', simple_value=reward)
                summary.value.add(tag='Perf/Reward_AVG', simple_value=reward_avg)
                self.summary_writer.add_summary(summary, episode)
                self.summary_writer.flush()

            return
            # save a snapshot of the actual graph & the buy/sell signals so we can visualize elsewhere
            if total > self.start_cap * 2:
                y = [float(p) for p in self.prices]
                signals = [int(sig) for sig in self.signals]
            else:
                y = None
                signals = None

            q = text("""
                insert into episodes (episode, reward, cash, value, agent_name, steps, y, signals) 
                values (:episode, :reward, :cash, :value, :agent_name, :steps, :y, :signals)
            """)
            conn.execute(q, episode=episode, reward=reward, cash=cash, value=value,
                         agent_name=self.agent_name, steps=self.timestep, y=y, signals=signals)


class BitcoinEnvTforce(OpenAIGym):
    def __init__(self, **kwargs):
        super(BitcoinEnvTforce, self).__init__('BTC-v0')
        if 'Nstep' not in kwargs['agent_name']:
            print('seeding')
            self.gym.env.seed(1234)
        self.gym.env.init(self.gym, **kwargs)


def scale_features_and_save():
    """
    If we want to scale/normalize or min/max features (states), first run the Env in log_states=True mode for a while,
    then call this function manually from python shell
    """
    observations = conn.execute('select obs from observations').fetchall()
    observations = [o[0] for o in observations]

    mat = np.array(observations)
    min_max = [np.floor(np.amin(mat, axis=0)), np.ceil(np.amax(mat, axis=0))]
    print('min/max: ', min_max)
    joblib.dump(min_max, 'data_/min_max.pkl')

    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(observations)
    joblib.dump(scaler, 'data_/scaler.pkl')

    mat = np.array(scaled)
    min_max = [np.floor(np.amin(mat, axis=0)), np.ceil(np.amax(mat, axis=0))]
    print('min/max (scaled): ', min_max)
    joblib.dump(min_max, 'data_/min_max_scaled.pkl')
