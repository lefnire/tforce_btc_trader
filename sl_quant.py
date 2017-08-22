import os
import numpy as np

import data

# np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

import pandas as pd
from tradingWithPython.lib.backtest import Backtest
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import SMA, RSI, ATR
from sklearn.externals import joblib

ACTION_BUY = 0
ACTION_SELL = 1
ACTION_HOLD = 2
ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]


if not os.path.exists('./plt'): os.makedirs('./plt')
if not os.path.exists('./data'): os.makedirs('./data')

# Load data
def read_convert_data():
    prices = data.db_to_dataframe(limit=6000) # FIXME need to implement streaming! Maxing out GPU
    prices.to_pickle('data/db_to_dataframe.pkl')
    return


def load_data():
    prices = pd.read_pickle('data/db_to_dataframe.pkl')
    # prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume (BTC)': 'volume'}, inplace=True)
    split = int(len(prices)*.8)
    x_train = prices.iloc[:split, ]
    x_test = prices.iloc[split:, ]
    return x_train, x_test


# Initialize first state, all items are placed deterministically
def init_state(indata, test=False):
    columns = []
    y_predict = indata[data.config.data.y_predict_column].values
    for k in data.tables:
        curr_indata = indata.rename(columns={
            k + '_last': 'close',
            k + '_high': 'high',
            k + '_low': 'low',
            k + '_volume': 'volume'
        })

        if not data.config.data.use_ta_indicators:
            columns += [
                curr_indata['close'].values,
                curr_indata['high'].values,
                curr_indata['low'].values,
                curr_indata['volume'].values
            ]
            continue

        # Aren't these features besides close "feature engineering", which the neural-net should do away with?
        close = curr_indata['close'].values
        diff = np.diff(close)
        diff = np.insert(diff, 0, 0)
        sma15 = SMA(curr_indata, timeperiod=15)
        sma60 = SMA(curr_indata, timeperiod=60)
        rsi = RSI(curr_indata, timeperiod=14)
        atr = ATR(curr_indata, timeperiod=14)
        columns += [close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr]

    # --- Preprocess data
    xdata = np.column_stack(columns)

    xdata = np.nan_to_num(xdata)
    if test:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    else:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    state = xdata[0:1, 0:1, :]

    return state, xdata, y_predict


# Take Action
def take_action(state, xdata, action, signal, time_step):
    """This should generate a list of trade signals that at evaluation time are fed to the backtester
    the backtester should get a list of trade signals and a list of price data for the asset"""

    # make necessary adjustments to state and then return it
    time_step += 1

    # if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == xdata.shape[0]:
        state = xdata[time_step - 1:time_step, 0:1, :]
        terminal_state = 1
        signal.loc[time_step] = 0

        return state, time_step, signal, terminal_state

    # move the market data window one step forward
    state = xdata[time_step - 1:time_step, 0:1, :]
    # take action
    if action == ACTION_BUY:
        signal.loc[time_step] = 5
    elif action == ACTION_SELL:
        signal.loc[time_step] = -5
    else:
        signal.loc[time_step] = 0
    terminal_state = 0
    # print("State: {}, signal: {}".format(state, signal))

    return state, time_step, signal, terminal_state


# Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)

    if not eval:
        bt = Backtest(
            pd.Series(
                data=[x for x in xdata[time_step - 2:time_step]],
                index=signal[time_step - 2:time_step].index.values
            ),
            signal[time_step - 2:time_step],
            signalType='shares'
        )
        reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2]) * bt.data['shares'].iloc[-1])

    if eval and terminal_state == 1:
        # save a figure of the test set
        bt = Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]
        plt.figure(figsize=(3, 4))
        bt.plotTrades()
        plt.axvline(x=400, color='black', linestyle='--')
        plt.text(250, 400, 'training data')
        plt.text(450, 400, 'test data')
        plt.suptitle(str(epoch))
        # plt.savefig('plt/' + str(epoch) + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
        plt.close('all')

    if time_step % 500 == 0:
        print(time_step, terminal_state, eval, reward)

    return reward


def evaluate_Q(eval_data, eval_model, price_data, epoch=0):
    # This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = init_state(eval_data)
    status = 1
    terminal_state = 0
    time_step = 1
    while status == 1:
        # We start in state S
        # Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        # Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        # Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True,
                                 epoch=epoch)
        state = new_state
        if terminal_state == 1:  # terminal state
            status = 0

    return eval_reward


# This neural network is the the Q-function, run it like this:
# model.predict(state.reshape(1,64), batch_size=1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

tsteps = 1
batch_size = 1
# 7 is the #features per table defined in load_data(). 4 w/o TA indicators (high, low, close, volume)
num_features = (7 if data.config.data.use_ta_indicators else 4) * len(data.tables)

model = Sequential()
model.add(LSTM(150,
               input_shape=(1, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.2))

model.add(LSTM(150,
               input_shape=(1, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.2))

model.add(Dense(len(ACTIONS), kernel_initializer='lecun_uniform'))
model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

# try "nadam", "rmsprop"
# TODO read http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
model.compile(loss='mse', optimizer="nadam")

import random, timeit

start_time = timeit.default_timer()

read_convert_data()  # run once to read indata, resample and convert to pickle
indata, test_data = load_data()
epochs = 200
gamma = 0.97  # since the reward can be several time steps away, make gamma high # TODO try higher, 99? (see book)
epsilon = 1
batchSize = 100
buffer = 200
replay = []
learning_progress = []
# stores tuples of (S, A, R, S')
h = 0
# signal = pd.Series(index=market_data.index)
signal = pd.Series(index=np.arange(len(indata)))
for i in range(epochs):
    if i == epochs - 1:  # the last epoch, use test data set
        state, xdata, price_data = init_state(test_data, test=True)
    else:
        state, xdata, price_data = init_state(indata)
    status = 1
    terminal_state = 0
    # time_step = market_data.index[0] + 64 #when using market_data
    time_step = 14
    # while game still in progress
    while status == 1:
        # We are in state S
        # Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state, batch_size=1)
        if random.random() < epsilon:  # choose random action
            action = np.random.randint(0, len(ACTIONS))
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))
        # Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        # Observe reward
        reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state)

        # Experience replay storage
        if len(replay) < buffer:  # if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
            # print(time_step, reward, terminal_state)
        else:  # if buffer full, overwrite old values
            if h < (buffer - 1):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            # randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                # Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state, batch_size=1)
                newQ = model.predict(new_state, batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1, len(ACTIONS)))
                y[:] = old_qval[:]
                if terminal_state == 0:  # non-terminal state
                    update = (reward + (gamma * maxQ))
                else:  # terminal state
                    update = reward
                y[0][action] = update
                # print(time_step, reward, terminal_state)
                X_train.append(old_state)
                y_train.append(y.reshape(len(ACTIONS), ))

            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=0)

            state = new_state
        if terminal_state == 1:  # if reached terminal state, update epoch status
            status = 0
    eval_reward = evaluate_Q(test_data, model, price_data, i)
    learning_progress.append((eval_reward))
    print("Epoch #: {} Reward: {} Epsilon: {}".format(i, eval_reward, epsilon))
    # learning_progress.append((reward))
    if epsilon > 0.1:  # decrement epsilon over time
        epsilon -= (1.0 / epochs)

elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in {}".format(elapsed))

bt = Backtest(pd.Series(data=[x[0, 0] for x in xdata]), signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

print(bt.data)
unique, counts = np.unique(filter(lambda v: v == v, signal.values), return_counts=True)
print(np.asarray((unique, counts)).T)

plt.figure()
plt.subplot(3, 1, 1)
bt.plotTrades()
plt.subplot(3, 1, 2)
bt.pnl.plot(style='x-')
plt.subplot(3, 1, 3)
plt.plot(learning_progress)
plt.show(block=True)

# FIXME
plt.savefig('plt/summary' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
plt.close('all')