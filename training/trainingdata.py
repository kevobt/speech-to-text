import errno
import json
import os

from os.path import exists

from typing import List

from speech.alphabet import Alphabet

from logging import getLogger

from utils.fs import safe_open


class TrainingData:
    def __init__(self, path: str, text: str, size: int):
        self.path = path
        self.text = text
        self.size = size

    def to_json(self):
        return {
            'path': self.path,
            'text': self.text,
            'size': self.size
        }


def validate(training_data: List[TrainingData], alphabet: Alphabet) -> List[TrainingData]:
    """
    Checks whether the text is element of the alphabet
    and whether the path to the audio file exists
    :param training_data: data to validate
    :param alphabet: alphabet which will be tested against the text of the training data
    :return: List of valid training data
    """
    log = getLogger(__name__)
    log.info('Validating training data ...')
    log.info('Items to validate: %s' % len(training_data))

    valid_training_data = []
    for data in training_data:
        if not is_text_valid(data.text, alphabet):
            continue
        if exists(data.path):
            valid_training_data.append(data)
        else:
            log.warning('The path to the training data "%s" does not exist' % data.path)

    log.info('Valid Items: %s' % len(valid_training_data))
    log.info('Validated training data')

    return valid_training_data


def is_text_valid(text: str, alphabet: Alphabet) -> bool:
    """
    Checks whether each character of the text lies inside the alphabet
    :param text: text to validate
    :param alphabet: alphabet containing the valid characters
    :return: Returns whether the character sequence is valid
    """
    for character in text:
        if character not in alphabet:
            return False

    return True


def load(path: str) -> List[TrainingData]:
    """
    Loads a file containing training data
    :param path: path to the file
    :return: List of training data
    """
    log = getLogger(__name__)
    log.info('Validating training data ...')

    if not exists(path):
        log.error('The training data file at "%s" does not exist' % path)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    with open(path, 'r') as file:
        training_data = [TrainingData(data['path'], data['text'], data['size']) for data in json.load(file)]

    return training_data
