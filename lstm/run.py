import os, time, h5py, sys
import gdax
import numpy as np
from btc import config
from btc.lstm import lstm, etl, plotting

tstart = time.time()
dl = etl.ETL()
true_values = []
all_predictions = []
holdings, wallet, dodged = 1000, 1000, 0


def percent_change(new, old):
    if old == 0: return .000000001
    return (new - old) / old


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def buy_sell(preds, trues, live=False):
    """Running talley of our experiment (how much we'll make)"""
    global wallet, holdings, dodged
    prev = ()
    for pred, true in zip(preds, trues):
        if not prev:
            prev = (pred, true); continue
        p_change, t_change = percent_change(pred, prev[0]), percent_change(true, prev[1])
        msg = 'Prediction={} ({}%); True={} ({}%) - '.format(pred, p_change, true, t_change)
        if p_change >= .001:
            msg = "BUYING! " + msg
            holdings += wallet
            wallet = 0
            if live:
                pass
                # auth_client.sell
        elif p_change <= -.001:
            msg = "SELLING! " + msg
            wallet += holdings
            dodged = dodged + holdings * t_change
            holdings = 0
            if live:
                pass
                # auth_client.buy(price='0.01',, size='0.01', product_id='BTC-USD') # USD, BTC, product
        if live: print(msg)
        holdings += holdings * t_change
    print('Holdings=${} Wallet=${} Dodged=${}'.format(holdings, wallet, dodged))
    if holdings + wallet < 0 and live:
        raise Exception("You died.")



if config.flags.create_clean_data or not os.path.isfile(config.data.filename_clean):
    print('> Generating clean data from:', config.data.filename_clean, 'with batch_size:',
          config.data.batch_size)
    dl.create_clean_datafile()

if config.flags.train or not os.path.isfile(config.model.filename_model):
    data_gen_train = dl.generate_clean_data()

    with h5py.File(config.data.filename_clean, 'r') as hf:
        nrows = hf['x'].shape[0]
        ncols = hf['x'].shape[2]

    ntrain = int(config.data.train_test_split * nrows)
    steps_per_epoch = int((ntrain / config.model.epochs) / config.data.batch_size)
    print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')

    model = lstm.build_network([ncols, 150, 150, 1])
    model.fit_generator(
        data_gen_train,
        steps_per_epoch=steps_per_epoch,
        epochs=config.model.epochs
    )
    model.save(config.model.filename_model)
    print('> Model Trained! Weights saved in', config.model.filename_model)

    data_gen_test = dl.generate_clean_data(start_index=ntrain)

    ntest = nrows - ntrain
    steps_test = int(ntest / config.data.batch_size)
    print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

    predictions = model.predict_generator(
        dl.generator_strip_xy(data_gen_test, true_values),
        steps=steps_test
    )

    # Save our predictions
    with h5py.File(config.model.filename_predictions, 'w') as hf:
        dset_p = hf.create_dataset('predictions', data=predictions)
        dset_y = hf.create_dataset('true_values', data=true_values)

    buy_sell(predictions, true_values)
    plotting.plot_results(predictions, true_values, block=True)

    # Show multi-window
    sys.exit(0)

    # Reload the data-generator
    data_gen_test = dl.generate_clean_data(
        batch_size=800,
        start_index=ntrain
    )
    data_x, true_values = next(data_gen_test)
    window_size = 50  # numer of steps to predict into the future

    # We are going to cheat a bit here and just take the next 400 steps from the testing generator and predict that
    # data in its whole
    predictions_multiple = predict_sequences_multiple(
        model,
        data_x,
        data_x[0].shape[0],
        window_size
    )

    plotting.plot_results_multiple(predictions_multiple, true_values, window_size)

else:
    # Initialize an array of 50 points, since current matplotlib code won't work with dynamic window-sizes. We'll
    # just modify this window inline and re-draw
    true_values = [0.] * 50
    all_predictions = [0.] * 50
    p_changes = [0.] * 50
    t_changes = [0.] * 50
    model = lstm.load_network()

    while True:
        # Market data will be running in the background (populate.py), so each pass will have new data
        x, y = dl.data_tail()

        true = float(y)
        true_values.pop(0);true_values.append(true)
        prediction = float(model.predict_on_batch(x)[0])
        all_predictions.pop(0);all_predictions.append(prediction)

        p_change = percent_change(prediction, all_predictions[-2])
        t_change = percent_change(true, true_values[-2])
        p_changes.pop(0);p_changes.append(p_change)
        t_changes.pop(0);t_changes.append(t_change)

        buy_sell(all_predictions[-2:], true_values[-2:], live=True)

        plotting.plot_results(p_changes, t_changes)
        time.sleep(5)
