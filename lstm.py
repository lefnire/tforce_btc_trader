import os, time, warnings
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from data import config

warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


def build_network(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("tanh"))

    start = time.time()
    model.compile(
        loss=config.model.loss_function,
        optimizer=config.model.optimiser_function)

    print("> Compilation Time : ", time.time() - start)
    return model


def load_network():
    """Load the h5 saved model and weights"""
    filename = config.model.filename_model
    if (os.path.isfile(filename)):
        return load_model(filename)
    else:
        print('ERROR: "' + filename + '" file does not exist as a h5 model')
        return None
