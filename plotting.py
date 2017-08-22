import matplotlib.pyplot as plt
from collections import namedtuple
from data import config

# If a plot exists, we'll update it
Existing = namedtuple('Existing', ['li_true', 'li_pred', 'fig'])
existing = None


def plot_results(predicted_data, true_data, block=False):
    global Existing, existing

    if existing:
        existing.li_true.set_ydata(true_data)
        existing.li_pred.set_ydata(predicted_data)
        existing.fig.canvas.draw()
    else:
        fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        if not config.flags.train:
            ax.set_ylim([-1, 1])
            ax.set_xlim([0, 50])
        li_true, = ax.plot(true_data, label='True Data')
        li_pred, = plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.show(block=block)
        existing = Existing(li_true=li_true, li_pred=li_pred, fig=fig)


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()