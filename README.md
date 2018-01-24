# TensorForce Bitcoin Trading Bot

A [TensorForce](https://github.com/reinforceio/tensorforce)-based Bitcoin trading bot (algo-trader). Uses deep reinforcement learning to automatically buy/sell/hold BTC based on price history.

Chat on [Gittr](https://gitter.im/lefnire/tforce_btc_trader)!

### 1. Setup
- Python 3.6+ (I use template strings a lot)
- Install & setup Postgres
  - Create two databases: `btc_history` and `hyper_runs`. You can call these whatever you want, and just use one db instead of two if you prefer (see Data section).
  - `cp config.example.json config.json`, pop ^ into `config.json`
- Install [TA-Lib](https://github.com/mrjbq7/ta-lib) manually.
- `pip install -r requirements.txt`
  - If issues, try installing these deps manually.

Note: you'll wanna run this on a beefy rig. Use a solid GPU, these are CNNs - minimum I've had success with is K80; 2-3x perf using 1080ti. Have lots of RAM available, my runs take 8GB+ (this could be greatly optimized the way step windows are buffered). Want my advice, build a PC or use Google Cloud Platform w/ Preemptible P100s.

### 2. Populate Data
- Download [mczielinski/bitcoin-historical-data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)
- Extract to `data/populate/bitcoin-historical-data`
- `python data/populate/kaggle.py`
- `python -c 'from data.data import setup_runs_table;setup_runs_table()'`
  - If you have trouble with that, just copy/paste the SQL from that file, execute against your `hyper_runs` DB from above.

### 3. Hypersearch
The crux of practical reinforcement learning is finding the right hyper-parameter combo (things like neural-network width/depth, L1 / L2 / Dropout numbers, etc). Some papers have listed optimal default hypers. Eg, the Proximate Policy Optimization (PPO) [paper](https://blog.openai.com/openai-baselines-ppo/) has a set of good defaults. But in my experience, they don't work well for our purposes (time-series / trading). I'll keep my own "best defaults" updated in this project, but YMMV and you'll very likely need to try different hyper combos yourself. The file `hypersearch.py` will search hypers forever, ever honing in on better and better combos (using Bayesian Optimization (BO), see `gp.py`). See Hypersearch section below for more details.

`python hypersearch.py`

Optional flags:
- `--guess <int>`: sometimes you don't want BO, which is pretty willy-nilly at first, to do the searching. Instead you want to try a hunch or two of your own first. Open `hypersearch.py` and add some hyper-override dicts in the `guess_overrides` array in `utils.py`, use `--guess <idx in that array>`. Using `--guess 0`, which is `{}` (aka no overrides) runs hypersearch against the hard-coded default hypers, but you might as well run `run.py` (below).
- `--gpu-split <int>`: number of ways to split a single GPU. Sometimes you'll be using a 1080ti or Tesla V100 - beastly GPUs - and `nvidia-smi -l` will show you're using some 10-20% of your GPU. What a waste. So you can use `--gpu-split 3` to split your V100 3 ways in 3 separate tabs, getting more bang-for-buck.
- `--net-type <lstm|conv2d>`: see discussion below (LSTM v CNN)
- `--boost`: you can optionally use gradient boosting when searching for the best hyper combo, instead of BO. BO is more exploratory and thorough, gradient boosting is more "find the best solution _now_". I tend to use `--boost` after say 100 runs are in the database, since BO may still be dilly-dallying till 200-300 and daylight's burning. Boost will suck in the early runs.

### 4. Run
Once you've found a good hyper combo from above (this could take days or weeks!), it's time to run your results.

`python run.py`

By itself it's not very useful (since you already have that run in the database), so augment with these args:

- `--id <int>`: the id of some winning hyper-combo you want to run with. Without this, it'll run from the hard-coded hyper defaults.
- `--gpu-split`: (see Hypersearch section)
- `--runs <int>`: `hypersearch.py` & `run.py` both have a max number of test runs (40 currently), you can increase or decrease that (maybe you think your run in the database could do better if given more time; increase the number here).
- `--live`: whooa boy, time to put your agent on GDAX and make real trades! I'm gonna let you figure out how to plug it in on your own, 'cause that's danger territory. I ain't responsible for shit. In fact, let's make that real - disclaimer at the end of README.
- `--live-test`: same as `live`, but without making the real trades. This will start monitoring a live-updated database (from config.json), same as `live`, but instead of making the actual trade, it pretends it did and reports back how much you would have made/lost. Dry-run. You'll definitely want to run this once or twice before running `--live`.
- `--early-stop <int>`: sometimes your models can overfit. In particular, PPO can give you great performance for a long time and then crash-and-burn. That kind of behavior will be obvious in your visualization (below), so you can tell your run to stop after x consecutive positive episodes (depends on the agent - some find an optimum and roll for 3 positive episodes, some 8, just eyeball your graph).

The result of `run.py` without `--live` or `--live-test` is to save the trained model to a directory (named `{id}{_early?}`, ie `10` or `10_early`). It'll then use that saved model when you run in `--live` or `--live-test` (use the same args, ie `--id 10 --early-stop 8` so it reconstructs the directory name).

## 5. Visualize
TensorForce comes pre-built with reward visualization on a TensorBoard. Check out their Github, you'll see. I needed much more customization than that for viz, so we're not using TensorBoard. I created a mini Flask server (2 routes) and a D3 React dashboard where you can slice & dice hyper combos, visualize progression, etc. If you click on a single run, it'll display a graph of the buy/sell signals that agent took in a time-slice (test-set) so you can eyeball whether he's being smart.

- Server: `cd visualize;FLASK_APP=server.py flask run`
- Client:
  - `cd visualize/client`
  - `npm install;npm install -g webpack-dev-server`
  - `npm start` => localhost:8080

![](https://s3.amazonaws.com/lefnirePublicFiles/adam_screenshot.png)

---

## About

This project is a [TensorForce](https://github.com/reinforceio/tensorforce)-based Bitcoin trading bot (algo-trader). It uses deep reinforcement learning to automatically buy/sell/hold BTC based on what it learns about BTC price history. Most blogs / tutorials / boilerplate BTC trading-bots you'll find out there use supervised machine learning, likely an LTSM. That's well and good - supervised learning learns what makes a time-series tick so it can predict the next-step future. But that's where it stops. It says "the price will go up next", but it doesn't tell you what to do. Well that's simple, buy, right? Ah, buy low, sell high - it's not that simple. Thousands of lines of code go into trading rules, "if this then that" style. Reinforcement learning takes supervised to the next level - it _embeds_ supervised within its architecture, and then decides what to do. It's beautiful stuff! Check out:

- [Sutton & Barto](http://amzn.to/2EWvnVf): de-facto textbook on RL basics
- [CS 294](http://rll.berkeley.edu/deeprlcourse/): the modern deep-learning spin on ^.

This project goes with Episode 26+ of [Machine Learning Guide](http://ocdevel.com/podcasts/machine-learning). Those episodes are tutorial for this project; including an intro to Deep RL, hyperparameter decisions, etc.


### Data
For this project I recommend using the Kaggle dataset described in Setup. It's a really solid dataset, best I've found! I'm personally using a friend's live-ticker DB. Unfortunately you can't. It's his personal thing, he may one day open it up as a paid API or something, we'll see. There's also some files in `data/populate` which use the CryptoWat.ch API. Great API _going forward_, but doesn't have the history you'll need to train on. If any y'all find anything better than the Kaggle set, LMK.

So here's how this project splits up databases (see `config.json`). We start with a `history` DB, which has all the historical BTC prices for multiple exchanges. Import it, train on it. Then we have an optionally separate `runs` database, which saves the results of each of your `hypersearch.py` runs. This data is used by our BO or Boost algo to search for better hyper combos. You can have `runs` table in your `history` database if you want, one-and-the-same. I have them separate because I want the `history` DB on localhost for performance reason (it's a major perf difference, you'll see), and `runs` as a public hosted DB, which allows me to collect runs from separate AWS p3.8xlarge running instances.

Then, when you're ready for live mode, you'll want a `live` database which is real-time, constantly collecting exchange ticker data. `--live` will handle keeping up with that database. Again, these can all 3 be the same database if you want, I'm just doing it my way for performance.

### LSTM v CNN
You'll notice the `--net-type <lstm|conv2d>` flag in `hypersearch.py` and `run.py`. This will select between an LSTM Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNN). I have them broken out of the hypersearch since they're so different, they kinda deserve their own `runs` DB each - but if someone can consolidate them into the hypersearch framework, please do. You may be thinking, "BTC prices is time-series, time-series is LSTM... why CNN?" It strangely turns out that LSTM doesn't do so hot here. In my own experience, in colleagues' experience, and in 2-3 papers I've read ([here's one](https://arxiv.org/pdf/1706.10059.pdf)) - we're all coming to the same conclusion. We're not sure why... the running theory is vanishing/exploding gradient. LSTMs work well in NLP which has some maximum 50-word sentences or so. LSTMs mitigated vanilla RNN's vanishing/exploding gradient for such sentences, true - but BTC history is infinite (on-going). Maybe LSTM can only go so far with time-series. Another possibility is that Deep Reinforcement Learning is most commonly researched, published, and open-sourced using CNNs. This because RL is super video-game centric, self-driving cars, all the vision stuff. So maybe the math behind these models lends better to CNNs? Who knows. The point is - experiment with both. Report back on Github your own findings.

So how does CNN even make sense for time-series? Well we construct an "image" of a time-slice, where the x-axis is time (obviously), the y-axis (height) is nothing... it's [1]. The z-axis (channels) is features (OHLCV, VWAP, bid/ask, etc). This is kinda like our agent literally looking at an image of price actions, like we do when day-trading, but a bit more robot-friendly / less human-friendly.

### Reinforcement Models

TensorForce has all sorts of models you can play with. This project currently only supports Proximate Policy Optimization (PPO), but I encourage y'all to add in other models (esp VPG, TRPO, DDPG, ACKTR, etc) and submit PRs. ACKTR is the current state-of-the-art Policy Gradient model, but not yet available in TensorForce. PPO is the second-most-state-of-the-art, so we're using that. TRPO is 3rd, VPG is old. DDPG I haven't put much thought into.

Those are the Policy Gradient models. Then there's the Q-Learning approaches (DQNs, etc). We're not using those because they only support discrete actions, not continuous actions. Our agent has one discrete action (buy|sell|hold), and one continuous action (how much?). Without that "how much" continuous flexibility, building an algo-trader would be... well, not so cool. You could do something like (discrete action = (buy-$200, sell-$200, hold)), but I dunno man... continuous is slicker.

### Hypersearch

You're likely familiar with _grid search_ and _random search_ when searching for optimial hyperparameters for machine learning models. Grid search searches literally every possible combo - exhaustive, but takes infinity years (especially w/ the number of hypers we work with in this project). Random search throws a dart at random hyper combos over and over, and you just kill it eventually and take the best. Super naive - it works ok for other ML setups, but in RL hypers are the make-or-break; more than model selection. Seriously, I've found L1 / L2 / Dropout selection more consequential than PPO vs DQN, LSTM vs CNN, etc.

That's why we're using Bayesian Optimization (BO). Or sometimes you'll hear Gaussian Processes (GP), the thing you're optimizing with BO. See `gp.py`. BO starts off like random search, since it doesn't have anything to work with; and over time it hones in on the best hyper combo using Bayesian inference. Super meta - use ML to find the best hypers for your ML - but makes sense. Wait, why not use RL to find the best hypers? We could (and I tried), but deep RL takes 10s of thousands of runs before it starts converging; and each run takes some 8hrs. BO converges much quicker. I've also implemented my own flavor of hypersearch via Gradient Boosting (if you use `--boost` during training); more for my own experimentation.

We're using `gp.py`, which comes from [thuijskens/bayesian-optimization](https://github.com/thuijskens/bayesian-optimization). It uses scikit-learn's in-built GP functions. I also considered dedicated BO modules, like GPyOpt. I found `gp.py` easier to work with, but haven't compared it's relative performance, nor its optimal hypers (yes, BO has its own hypers... it's turtles all the way down. But luckily I hear you can pretty safely use BO's defaults). If anyone wants to explore any of that territory, please indeed!

### License: AGPLv3.0

GPL bit so we share our findings. Community effort, right? Boats and tides. Affero bit so we can all run our own trading instances w/ personal configs / mods. Heck, any of us could run this as a service / hedge fund. I'm pretty keen on this license, having used it in a prior [internet company](https://habitica.com) I'd founded; but if someone feels strongly about a different license, please open an issue & LMK - open to suggestions. See LICENSE.

### Disclaimer

By using this code you accept all responsibility for money lost because of this code.