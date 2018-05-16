import os
import json

from typing import List

from logger import get_logger


class TrainingStatistics:
    def __init__(self, loss: List[float], validation_loss: List[float]):
        self.loss = loss
        self.validation_loss = validation_loss

    def to_json(self):
        return {
            "loss": self.loss,
            "validationLoss": self.validation_loss,
        }

    @property
    def current_epoch(self):
        # after each epoch the training loss will be stored.
        # therefore, the amount of available losses is equal to the amount of passed epochs.
        return len(self.loss)


def load(path: str) -> TrainingStatistics:
    """
    Gets the history of a training containing the losses
    :param path: path to the history of the training
    :return: TrainingStatistics
    """
    with open(path, 'r') as file:
        training_plan = json.load(file)

    return TrainingStatistics(training_plan["loss"],
                              training_plan["validationLoss"])


def save(path: str, statistics: TrainingStatistics):
    log = get_logger(__name__)

    with open(path, 'w') as file:
        file.write(json.dumps(statistics.to_json(), indent=4))
    log.info('Saved statistics to %s' % os.path.abspath(path))
