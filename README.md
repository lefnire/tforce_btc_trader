# BTC-trading Reinforcement Learning (RL) Bot
Uses reinforcement learning to learn when to buy/sell/hold based on reward/punishment. Of course, reward/punishment can be long-term ("the credit assignment problem") and that's where RL comes in with Deep Q-Network (DQN) methods (DQN, DDQN, etc) and Policy Gradient methods (PPO, TRPO, etc). In our case we'll be acting past information complexly and hierarchically (ie, the Markov Property doesn't hold here as it does in many image-based video games where RL is applied). Hence our learned Policy or Q-function will be through an LSTM RNN rather than an MLP/CNN. We might be able to turn this into a Markov approach and use an MLP by using _indicators_ as features in every state (SMA, RSI, ATR) - this is something Tyler will experiment with in coming days.

Currently the agent isn't working very well. It converges on some actions whose rewards put him at ~$150, having started with $200. Through various hyperparameter-combo-tuning, I've increased it inch-by-inch, but he's still not learning to make money. Once he can consistently make any amount of profit, then I'll pull you @Alex into the Backtesting bit.

## TODO:
* King-of-the-hill hyperparamater tuning to find the best combo. Current running winners/losers in tforce_run.py comments.
* Try TA-Lib indicators and "dense" layers, Markov style. Try literally viewing the BTC graph as an image, use CNNs the way most Gym/TensorForce environments are used.
* Implement proper Backtesting (see PyAlgoTrade, Backtrader, Zipline, etc). Currently I'm doing a very naive test: up/down percent on any $ you have in your portfolio. Reward/punishment is simple $-gained / $-lost, but eventually we'll want a more robust PNL/Sharpe/etc system. I just want to prove RL an work period first (it's not working yet).

# Original attempts (see git history)
* [sl-quant](https://medium.com/@danielzakrisson/the-self-learning-quant-d3329fcc9915#.3b4ghaoa7) RL-based, but the DQN logic was custom-coded and poorly-optimized, resulting in multi-day training for little progress. I may revisit, since it's pretty transparent and slim.
* [Multidimensional LSTM BitCoin Time Series](http://www.jakob-aungiers.com/articles/a/Multidimensional-LSTM-Networks-to-Predict-Bitcoin-Price) great progress for ticker _prediction_ via Supervised Learning (SL), but not action prediction via RL. Would need to hand-code trading strategies.
* [handson-ml](https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb) Supervised (see above).

# Setup
(I'll flesh this out & create a requirements.txt soon). Install Anaconda, create a Python3 environment. Matplotlib, Scikit-Learn, Pandas, Numpy, TensorFlow, the works. The unique requirements for this project are [TensorForce](https://github.com/reinforceio/tensorforce#installation) and [TA-Lib](https://github.com/mrjbq7/ta-lib#installation).