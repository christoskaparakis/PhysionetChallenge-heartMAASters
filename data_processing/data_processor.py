from scipy import signal
from helper_code import *
import numpy as np, os, joblib
import math
import wfdb
from wfdb import processing
from random import randint
import sys


def remove_baseline(recording, fs):
    b, a = signal.cheby2(3, 20, [0.5 / (fs / 2), 100 / (fs / 2)], btype='bandpass')
    signal_out = signal.filtfilt(b, a, recording)
    return signal_out


def remove_interference(recording, fs):
    omega = 2 * math.pi * 50 / fs;
    z1 = math.e ** (1j * omega)
    z2 = math.e ** (-1j * omega)
    p1 = 0.9 * z1
    p2 = 0.9 * z2

    b = [1, -(z1 + z2), z1 * z2]
    a = [1, -(p1 + p2), p1 * p2]

    signal_out = signal.filtfilt(b, a, recording)
    return signal_out


def resampling_frequency(record_name):
    record = wfdb.rdrecord(record_name.replace(".mat", ""))

    rec = np.zeros((int(record.sig_len / record.fs * 500), 12))

    if record.__dict__['fs'] != 500:
        for i in range(0, record.p_signal.shape[1]):
            rec[:, i], pos = wfdb.processing.resample_sig(record.p_signal[:, i], record.fs, 500)
        return rec
    else:
        return record.p_signal


def process_data(header, recording, leads):
    if np.any(np.isnan(recording)):
        where_are_NaNs = np.isnan(recording)
        recording[where_are_NaNs] = 0

    # Re Order leads
    available_leads = get_leads(header)

    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)

    recording = recording[:, indices].T
    num_leads = len(leads)
    recording = recording.astype("float")

    for i in range(num_leads):
        try: #catching ValueError exception produced in filtfilt
            temp = remove_baseline(recording[i, :], 500)
            temp = remove_interference(temp, 500)
            recording[i, :] = temp
        except ValueError:
            pass

    # calculating the amount of subparts and looping trough their indices
    new_recordings = np.zeros((len(leads), 5000))
    for i, lead in enumerate(recording):
        padding = np.array([0] * max(5000 - len(lead), 0))
        lead_recording = lead[0: min(len(lead), 5000)]
        q = np.concatenate((lead_recording, padding))

        new_recordings[i] = q

    return new_recordings.T


def labels_to_numberline(labels, unique_labels):
    label_am = len(unique_labels)
    _labels = np.zeros((label_am))
    for i in range(len(labels)):
        for j in range(len(unique_labels)):
            if labels[i] == unique_labels[j]:
                _labels[j] = 1

    return _labels
