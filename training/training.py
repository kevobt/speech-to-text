import os
import json
from itertools import chain

from typing import List

from keras.models import Model

from speech.alphabet import Alphabet
from speech.alphabet import load as load_alphabet

from training.trainingdata import TrainingData
from training.trainingconfig import TrainingConfig
from training.trainingconfig import load as load_config
from training.trainingdata import load as load_training_data, validate as validate_training_data

from models import get as load_model

from logger import get_logger

from utils.fs import safe_open

WEIGHTS_FILE_NAME = 'weights'


class Training:
    def __init__(self,
                 model: Model,
                 alphabet: Alphabet,
                 batch_size: int,
                 epochs: int,
                 loss: List[float],
                 validation_loss: List[float],
                 training_data: List[TrainingData]):
        self.model = model
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.validation_loss = validation_loss
        self.training_data = training_data

    def to_json(self):
        return {
            "loss": self.loss,
            "validationLoss": self.validation_loss,
            "alphabet": self.alphabet
        }

    @property
    def passed_epochs(self):
        return len(self.loss)


def load(path: str) -> Training:
    """
    Loads a training from an existing directory
    :param path: Path to the directory where the training data is stored
    :return: A training
    """
    log = get_logger(__name__)
    log.info('Loading training %s ...' % os.path.abspath(path))

    training_name = os.path.splitext(os.path.basename(path))[0]
    directory_path = os.path.dirname(path)
    weights_path = os.path.join(directory_path, training_name + '.weights.h5')
    training_config_path = os.path.join(directory_path, training_name + '.conf.json')
    save_file = os.path.join(directory_path, training_name + '.sav')

    # Check paths
    if not os.path.exists(save_file):
        raise FileNotFoundError('The training file "%s" does not exist' % save_file)

    if not os.path.exists(weights_path):
        raise FileNotFoundError('The weights file "%s" for this training is missing' % weights_path)

    if not os.path.exists(training_config_path):
        raise FileNotFoundError('The configuration file "%s" for this training is missing' % training_config_path)

    config = load_config(training_config_path)

    # load all training data like specified in the config
    try:
        training_data = list(chain.from_iterable([load_training_data(path) for path in config.training_data]))
    except FileNotFoundError as ex:
        log.error('Check the path of the trainingData property in your training configuration file')
        raise ex

    # limit data like specified in the config
    training_data = training_data[:config.training_data_quantity]

    # only use valid training data
    alphabet = load_alphabet(config.alphabet_path)
    training_data = validate_training_data(training_data, alphabet)

    with open(save_file, 'r') as file:
        training_save = json.load(file)

    model = load_model(config.net)
    batch_size = config.batch_size
    epochs = config.epochs
    loss = training_save['loss']
    validation_loss = training_save['validationLoss']

    model.load_weights(weights_path)

    log.info('Training loaded')

    return Training(model, alphabet, batch_size, epochs, loss, validation_loss, training_data)


def create(path: str, config: TrainingConfig, overwrite: bool) -> Training:
    """
    Creates a training object using the given configuration
    :param path: Path to the training directory
    :param config: Training plan
    :param overwrite: If true, existing training data will be overwritten
    :return: Training
    """
    log = get_logger(__name__)
    log.info('Preparing training ...')

    training_name = os.path.splitext(os.path.basename(path))[0]
    directory_path = os.path.dirname(path)
    save_path = os.path.join(directory_path, training_name + '.sav')
    weights_path = os.path.join(directory_path, training_name + '.weights.h5')
    training_config_path = os.path.join(directory_path, training_name + '.conf.json')

    # check if other training files could be overwritten
    if not overwrite:
        if os.path.exists(save_path) or os.path.exists(weights_path) or os.path.exists(training_config_path):
            raise FileExistsError('The training file "%s" already exists' % save_path)

    # copy the training configuration to the training directory
    with safe_open(training_config_path, "w") as file:
        file.write(json.dumps(config.to_json()))
        log.info('Copied training configuration to "%s"' % os.path.abspath(training_config_path))

    # create data for training object
    model = load_model(config.net)
    batch_size = config.batch_size
    epochs = config.epochs

    # load all training data like specified in the config
    try:
        training_data = list(chain.from_iterable([load_training_data(path) for path in config.training_data]))
    except FileNotFoundError as ex:
        log.error('Check the path of the trainingData property in your training configuration file')
        raise ex

    # limit data like specified in the config
    training_data = training_data[:config.training_data_quantity]

    # only use valid training data
    alphabet = load_alphabet(config.alphabet_path)
    training_data = validate_training_data(training_data, alphabet)

    log.info("Created training")

    return Training(model, alphabet, batch_size, epochs, [], [], training_data)


def save(path: str, training: Training, save_weights: bool):
    log = get_logger(__name__)
    training_name = os.path.splitext(os.path.basename(path))[0]
    directory_path = os.path.dirname(path)
    save_path = os.path.join(directory_path, training_name + '.sav')
    weights_path = os.path.join(directory_path, training_name + '.weights.h5')

    # save weights as .h5
    if save_weights:
        training.model.save_weights(weights_path)

    # save training .json
    with safe_open(save_path, "w") as file:
        file.write(json.dumps(training.to_json(), indent=4))

    log.info('Saved training to "%s' % os.path.abspath(save_path))
