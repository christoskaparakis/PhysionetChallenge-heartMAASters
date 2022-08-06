import numpy as np
from tensorflow import keras
from helper_code import get_labels, load_header, load_recording
from data_processing.data_processor import process_data, labels_to_numberline, resampling_frequency


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, uniq_labels, recording_filenames, head_filenames, leads, batch_size=32, dim=(5000,12), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.uniq_labels = uniq_labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.leads = leads

        self.record_names = recording_filenames
        self.head_names = head_filenames

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, len(self.uniq_labels)), dtype=int)
        # Generate data
        for i, ID in enumerate(list_ids):
            # Store sample
            header = load_header(self.head_names[ID])
            recording = resampling_frequency(self.record_names[ID])

            X[i,] = process_data(header, recording, self.leads)
            # Store class
            y[i] = labels_to_numberline(get_labels(header), self.uniq_labels)

        return X, y
