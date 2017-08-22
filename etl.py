import h5py, requests, time
import numpy as np
import pandas as pd
import data as btc_data
from data import config, conn, conn_btc


class ETL:
    """Extract Transform Load class for all data operations pre model inputs. Data is read in generative way to allow
    for large datafiles and low memory utilisation"""

    def __init__(self):
        self.btc_data = None

    def generate_clean_data(self, filename=config.data.filename_clean, batch_size=config.data.batch_size,
                            start_index=0):
        with h5py.File(filename, 'r') as hf:
            i = start_index
            while True:
                data_x = hf['x'][i:i + batch_size]
                data_y = hf['y'][i:i + batch_size]
                i += batch_size
                yield (data_x, data_y)

    def generator_strip_xy(self, data_gen, true_values):
        for x, y in data_gen:
            true_values += list(y)
            yield x

    def create_clean_datafile(self, batch_size=config.data.batch_size, x_window_size=config.data.x_window_size,
                              y_window_size=config.data.y_window_size):
        """Incrementally save a datafile of clean data ready for loading straight into model"""
        print('> Creating x & y data files...')

        data_gen = self.clean_data(
            batch_size=batch_size,
            x_window_size=x_window_size,
            y_window_size=y_window_size,
            y_col=config.data.y_predict_column
        )

        i = 0
        with h5py.File(config.data.filename_clean, 'w') as hf:
            x1, y1 = next(data_gen)
            # Initialise hdf5 x, y datasets with first chunk of data
            rcount_x = x1.shape[0]
            dset_x = hf.create_dataset('x', shape=x1.shape, maxshape=(None, x1.shape[1], x1.shape[2]), chunks=True)
            dset_x[:] = x1
            rcount_y = y1.shape[0]
            dset_y = hf.create_dataset('y', shape=y1.shape, maxshape=(None,), chunks=True)
            dset_y[:] = y1

            for x_batch, y_batch in data_gen:
                # Append batches to x, y hdf5 datasets
                print('> Creating x & y data files | Batch:', i, end='\r')
                dset_x.resize(rcount_x + x_batch.shape[0], axis=0)
                dset_x[rcount_x:] = x_batch
                rcount_x += x_batch.shape[0]
                dset_y.resize(rcount_y + y_batch.shape[0], axis=0)
                dset_y[rcount_y:] = y_batch
                rcount_y += y_batch.shape[0]
                i += 1

        print('> Clean datasets created in file `' + config.data.filename_clean)

    def clean_data(self, batch_size, x_window_size, y_window_size, y_col):
        """Cleans and normalizes the data in batches `batch_size` at a time"""
        # data = self.db_to_dataframe(sklearn_normalize=not normalize)
        data = btc_data.btc_to_dataframe()

        # Convert y-predict column name to numerical index
        y_col = list(data.columns).index(y_col)

        num_rows = len(data)
        x_data = []
        y_data = []
        i = 0
        while (i + x_window_size + y_window_size) <= num_rows:
            x_window_data = data[i:(i + x_window_size)]
            y_window_data = data[(i + x_window_size):(i + x_window_size + y_window_size)]

            # Remove any windows that contain NaN
            if (x_window_data.isnull().values.any() or y_window_data.isnull().values.any()):
                i += 1
                continue

            if not config.data.sklearn_normalize:
                abs_base, x_window_data = self.zero_base_standardise(x_window_data)
                _, y_window_data = self.zero_base_standardise(y_window_data, abs_base=abs_base)

            # Average of the desired predicter y column
            y_average = np.average(y_window_data.values[:, y_col])
            x_data.append(x_window_data.values)
            y_data.append(y_average)
            i += 1

            # Restrict yielding until we have enough in our batch. Then clear x, y data for next batch
            if i % batch_size == 0:
                # Convert from list to 3 dimensional numpy array [windows, window_val, val_dimension]
                x_np_arr = np.array(x_data)
                y_np_arr = np.array(y_data)
                x_data = []
                y_data = []
                yield (x_np_arr, y_np_arr)


    def data_tail(self, x_window_size=config.data.x_window_size,
                  y_window_size=config.data.y_window_size):
        """Returns one batch worth of data (one x-window & y-window)"""
        data = btc.db_to_dataframe(tail=True)
        data = data[-(x_window_size + y_window_size):]  # last batch

        # Convert y-predict column name to numerical index
        y_col = list(data.columns).index(config.data.y_predict_column)

        x_window_data = data[:x_window_size]
        y_window_data = data[x_window_size:]

        if not config.data.sklearn_normalize:
            abs_base, x_window_data = self.zero_base_standardise(x_window_data)
            _, y_window_data = self.zero_base_standardise(y_window_data, abs_base=abs_base)

        # Average of the desired predicter y column
        y_average = np.average(y_window_data.values[:, y_col])

        # Convert from list to 3 dimensional numpy array [windows, window_val, val_dimension]
        x_data = np.array([x_window_data.values])
        y_data = np.array([y_average])
        return (x_data, y_data)

    def zero_base_standardise(self, data, abs_base=pd.DataFrame()):
        """Standardise dataframe to be zero based percentage returns from i=0"""
        if (abs_base.empty): abs_base = data.iloc[0]
        data_standardised = (data / abs_base) - 1
        return (abs_base, data_standardised)

    def min_max_normalize(self, data, data_min=pd.DataFrame(), data_max=pd.DataFrame()):
        """normalize a Pandas dataframe using column-wise min-max normalisation (can use custom min, max if desired)"""
        if (data_min.empty): data_min = data.min()
        if (data_max.empty): data_max = data.max()
        data_normalized = (data - data_min) / (data_max - data_min)
        return (data_min, data_max, data_normalized)
