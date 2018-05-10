from python_speech_features import mfcc

import scipy.io.wavfile as wavfile
import numpy as np


def calculate_mfccs(audio_file_path):
    """
    For a given audio clip, calculate the corresponding feature

    :param audio_file_path:
    :return: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    (rate, data) = wavfile.read(audio_file_path)
    return mfcc(data, rate, numcep=26)


def normalize_mfcc(feature, window, max_freq, eps=1e-14):
    """ Center a feature using the mean and std
    Params:
        feature (numpy.ndarray): Feature to normalize
    """
    feat_dim = calculate_feature_dimension(window, max_freq)
    feats_mean = np.zeros((feat_dim,))
    feats_std = np.ones((feat_dim,))

    return (feature - feats_mean) / (feats_std + eps)


def calculate_feature_dimension(window, max_freq):
    return int(0.001 * window * max_freq) + 1
