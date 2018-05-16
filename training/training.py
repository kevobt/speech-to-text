import errno
import os

from typing import List

from keras.models import Model

from speech.alphabet import Alphabet
from speech.alphabet import load as load_alphabet

from training.trainingdata import TrainingData
from training.trainingconfig import load as load_config
from training.trainingdata import load as load_training_data, validate as validate_training_data

from models import load as load_model_save_file, get as get_model

from logger import get_logger
from training.trainingstatistics import TrainingStatistics, load as load_statistics


class Training:
    def __init__(self,
                 model: Model,
                 alphabet: Alphabet,
                 batch_size: int,
                 epochs: int,
                 training_statistics: TrainingStatistics,
                 training_data: List[TrainingData]):
        self.model = model
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.epochs = epochs
        self.training_statistics = training_statistics
        self.training_data = training_data

    @property
    def passed_epochs(self):
        return len(self.training_statistics.loss)


def load(model_save_file_path: str,
         weights_path: str,
         training_configuration_path: str,
         training_statistics_path: str) -> Training:
    """
    Loads a training from an existing directory
    :param path: Path to the directory where the training data is stored
    :return: A training
    """
    log = get_logger(__name__)
    log.info('Loading training ...')

    # Check paths
    if not os.path.exists(model_save_file_path):
        raise FileNotFoundError('The model file "%s" does not exist' % model_save_file_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError('The weights file "%s" for this training is missing' % weights_path)

    if not os.path.exists(training_configuration_path):
        raise FileNotFoundError('The configuration file "%s" for this training is missing' % training_configuration_path)

    if not os.path.exists(training_statistics_path):
        raise FileNotFoundError('The configuration file "%s" for this training is missing' % training_statistics_path)

    config = load_config(training_configuration_path)

    # load all training data like specified in the config
    try:
        training_data = load_training_data(config.training_data)
    except FileNotFoundError as ex:
        log.error("Please check your configuration file at %s" % os.path.abspath(training_configuration_path))
        raise ex

    # load model from the save file
    model_save = load_model_save_file(model_save_file_path, weights_path)

    # only use valid training data and set to limitation defined in config
    training_data = validate_training_data(training_data, model_save.alphabet)
    training_data = training_data[:config.training_data_quantity]

    batch_size = config.batch_size
    epochs = config.epochs
    statistics = load_statistics(training_statistics_path)

    log.info('Training loaded')

    return Training(model_save.model, model_save.alphabet, batch_size, epochs, statistics, training_data)


def create(training_configuration_path: str) -> Training:
    """
    Creates a training object using the given configuration
    :param training_configuration_path: Training plan
    :return: Training
    """
    log = get_logger(__name__)
    log.info('Preparing training ...')

    #check paths
    if not os.path.exists(training_configuration_path):
        log.error('The configuration file at "%s" does not exist' % os.path.abspath(training_configuration_path))
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), training_configuration_path)

    config = load_config(training_configuration_path)

    # create data for training object
    model = get_model(config.net)
    batch_size = config.batch_size
    epochs = config.epochs

    # load all training data like specified in the config
    try:
        training_data = load_training_data(config.training_data)
    except FileNotFoundError as ex:
        log.error("Please check your configuration file at %s" % os.path.abspath(training_configuration_path))
        raise ex

    alphabet = load_alphabet(config.alphabet_path)

    # only use valid training data and set to limitation defined in config
    training_data = validate_training_data(training_data, alphabet)
    training_data = training_data[:config.training_data_quantity]

    log.info("Created training")

    return Training(model, alphabet, batch_size, epochs, TrainingStatistics([], []), training_data)

