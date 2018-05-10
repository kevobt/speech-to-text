import json

from typing import List


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


def load(file_name: str) -> TrainingStatistics:
    """
    Gets the history of a training containing the losses
    :param file_name: path to the history of the training
    :return: TrainingStatistics
    """
    with open(file_name, 'r') as file:
        training_plan = json.load(file)

    return TrainingStatistics(training_plan["loss"],
                              training_plan["validationLoss"])
