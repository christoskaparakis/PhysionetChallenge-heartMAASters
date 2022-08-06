#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.
from helper_code import *
import numpy as np, os, joblib

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, Input, Dense, concatenate, InputLayer
from tensorflow.keras.models import  Model
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras import optimizers
import _pickle as cPickle
from data_processing.generator import DataGenerator
from data_processing.data_processor import process_data
twelve_lead_model_filename = '12_lead_model.h5'
six_lead_model_filename = '6_lead_model.h5'
three_lead_model_filename = '3_lead_model.h5'
two_lead_model_filename = '2_lead_model.h5'

twelve_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
six_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
four_leads = ["I", "II", "III", "V2"]
three_leads = ["I", "II", "V2"]
two_leads = ["I", "II"]

scored = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003',
          '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007', '111975006',
          '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000', '63593006',
          '164934002', '59931005', '17338001']


# Converts labels of a recording to a binary array



# Converts a binary array eg [0,1,0,1] to the actual labels
def numberline_to_labels(labels, unique_labels):
    label_am = len(unique_labels)
    _labels = []
    for i in range(labels.shape[0]):
        label_list = []
        for j in range(label_am):
            if labels[i][j] == 1:
                label_list.append(unique_labels[j])
        _labels.append(label_list)
    return _labels


def setup_combined_model(label_am, lead_am):
    # Left side sub-model (Raw ECG data):
    L1 = Input(shape=(lead_am, 5000))
    L2 = Conv1D(32, kernel_size=5, activation='relu')(L1)
    L2 = layers.MaxPool1D()(L2)
    L2 = layers.Conv1D(32, kernel_size=5, activation='relu')(L2)
    L2 = layers.MaxPool1D()(L2)
    L2 = layers.Conv1D(64, kernel_size=5, activation='relu')(L2)
    L2 = layers.MaxPool1D()(L2)
    L2 = layers.Conv1D(64, kernel_size=5, activation='relu')(L2)
    L2 = layers.MaxPool1D()(L2)
    L2 = layers.Conv1D(128, kernel_size=5, activation='relu')(L2)
    L2 = layers.MaxPool1D()(L2)
    L2 = layers.Dropout(0.3)(L2)
    L2 = layers.Conv1D(128, kernel_size=4, activation='relu')(L2)
    L2 = layers.MaxPool1D()(L2)
    L2 = layers.Dropout(0.3)(L2)
    L2 = layers.Conv1D(128, kernel_size=4, activation='relu')(L2)

    L2 = layers.Flatten()(L2)
    L2 = layers.Dense(256, activation='relu')(L2)

    # Right side sub-model (Features):
    R1 = Input(shape=(lead_am, 1000))
    R2 = Dense(2, activation='softmax')(R1)
    # TODO : extra layers to feature model
    # Combining them together:
    merge = concatenate([L2, R2])

    # Simple feedforward layers with the combined layers
    hidden = Dense(300, activation='relu')(merge)
    hidden = Dense(200, activation='relu')(hidden)
    hidden = Dense(150, activation='relu')(hidden)

    # Output Layer:
    output = Dense(label_am, activation='sigmoid')(hidden)

    # Defining the model:
    model = Model(inputs=[L1, R1], outputs=output)

    return model


def setup_model(label_am, lead_am):
    # Create new model
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=(5000, lead_am)))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Conv1D(128, kernel_size=5, activation='relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(128, kernel_size=4, activation='relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(128, kernel_size=4, activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(label_am, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    return model


################################################################################
#
# Training function
#
################################################################################

def get_uniq_labels(all_headers):
    uniq_labels = set()
    for head_file in all_headers:
        header = load_header(head_file)
        labels = get_labels(header)
        for label in labels:
            uniq_labels.add(label)
    uniq_labels = list(uniq_labels)
    return uniq_labels


# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    head, record = find_challenge_files(data_directory)
    num_recordings = len(record)
    uniq_labels = get_uniq_labels(head)
    print("got recordings")
    if not num_recordings:
        raise Exception('No data was provided.')

    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    ids = list(range(0, num_recordings))

    params = {'dim': (5000, 12),
              'list_IDs' : ids,
              'uniq_labels': uniq_labels,
              'recording_filenames': record,
              'head_filenames':head,
              'batch_size': 128,
              'shuffle': True}

    epoch_am = 10
    # Train 12 lead model
    print("Train 12 lead model")
    model = setup_model(len(uniq_labels), 12)
    training_generator = DataGenerator(leads=twelve_leads, **params)

    model.fit(training_generator, epochs=epoch_am)

    save_model(model_directory, twelve_leads, uniq_labels, model)

    # Train 6 lead model
    print("Train 6 lead model")
    params['dim'] = (5000, 6)
    training_generator = DataGenerator(leads=six_leads, **params)

    model = setup_model(len(uniq_labels), 6)
    model.fit(training_generator, epochs=epoch_am)
    save_model(model_directory, six_leads, uniq_labels, model)

    # Train 4 lead model
    print("Train 4 lead model")
    params['dim'] = (5000, 4)
    training_generator = DataGenerator(leads=four_leads, **params)

    model = setup_model(len(uniq_labels), 4)
    model.fit(training_generator, epochs=epoch_am)

    save_model(model_directory, four_leads, uniq_labels, model)

    # Train 3 lead model
    print("Train 3 lead model")
    params['dim'] = (5000, 3)
    training_generator = DataGenerator(leads=three_leads, **params)

    model = setup_model(len(uniq_labels), 3)
    model.fit(training_generator, epochs=epoch_am)
    save_model(model_directory, three_leads, uniq_labels, model)

    # Train 2 lead model
    print("Train 2 lead model")
    params['dim'] = (5000, 2)
    training_generator = DataGenerator(leads=two_leads, **params)

    model = setup_model(len(uniq_labels), 2)
    model.fit(training_generator, epochs=epoch_am)

    save_model(model_directory, two_leads, uniq_labels, model)


def get_not_corresponding_label_ids(labels):
    not_in_twelve_leads = []
    for i in range(len(twelve_leads)):
        if twelve_leads[i] not in labels:
            not_in_twelve_leads.append(i)
    return not_in_twelve_leads

################################################################################
#
# File I/O functions
#
################################################################################

def save_model(model_directory, leads, classes, classifier):
    filename = os.path.join(model_directory, get_model_filename(leads))
    d = {'leads': leads, 'classes': classes, 'classifier': filename + '.h5'}

    classifier.save(filename + '.h5')
    joblib.dump(d, filename, protocol=0)


def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    d = joblib.load(filename)

    d['classifier'] = load_keras_model(d['classifier'])
    return d


def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_' + '-'.join(sorted_leads) + '.sav'

################################################################################
#
# Running trained model functions
#
################################################################################


# Generic function for running a trained model.
def run_model(model, header, recording):
    classifier = model['classifier']
    classes = model['classes']
    leads = model['leads']

    input = process_data(header, recording, leads)

    predictions = classifier.predict(np.expand_dims(input, axis=0))[0]
    # Set which labels are scored based on a cutoff percentage # TODO : ROC curve
    pred_labels = np.copy(predictions)
    for i in range(len(pred_labels)):
        if pred_labels[i] > 0.6:
            pred_labels[i] = 1
        else:
            pred_labels[i] = 0
            pred_labels[i] = 0
    pred_labels = pred_labels.astype('int')

    return classes, pred_labels, predictions
