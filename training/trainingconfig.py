import os
import json

from typing import List

from logger import get_logger
from utils.fs import safe_open


class TrainingConfig:
    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 net: str,
                 training_data_quantity: int,
                 training_data: List[str],
                 alphabet_path: str):
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = net
        self.training_data_quantity = training_data_quantity
        self.training_data = training_data
        self.alphabet_path = alphabet_path

    def to_json(self):
        return {
            'epochs': self.epochs,
            'batchSize': self.batch_size,
            'trainingDataQuantity': self.training_data_quantity,
            'net': self.net,
            'trainingData': self.training_data,
            'alphabetPath': self.alphabet_path
        }


def load(training_plan_path: str) -> TrainingConfig:
    if not os.path.exists(training_plan_path):
        raise FileNotFoundError('The training plan "%s" does not exist' % training_plan_path)
    with open(training_plan_path, 'r') as file:
        training_plan = json.load(file)

    return TrainingConfig(training_plan["epochs"],
                          training_plan["batchSize"],
                          training_plan["net"],
                          training_plan["trainingDataQuantity"],
                          training_plan["trainingData"],
                          training_plan["alphabetPath"])


def save(path: str, config: TrainingConfig):
    log = get_logger(__name__)

    with safe_open(path, 'w') as file:
        file.write(json.dumps(config.to_json(), indent=4))
    log.info('Saved config to %s' % os.path.abspath(path))
