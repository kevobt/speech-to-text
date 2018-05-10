import os
import json

from typing import List


class TrainingPlan:
    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 net: str,
                 training_data_quantity: int,
                 training_data: List[str]):
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = net
        self.training_data_quantity = training_data_quantity
        self.training_data = training_data

    def to_json(self):
        return {
            "epochs": self.epochs,
            'batchSize': self.batch_size,
            'trainingDataQuantity': self.training_data_quantity,
            'net': self.net,
            'trainingData': self.training_data
        }


def load(training_plan_path: str) -> TrainingPlan:
    if not os.path.exists(training_plan_path):
        raise FileNotFoundError('The training plan "%s" does not exist' % training_plan_path)
    with open(training_plan_path, 'r') as file:
        training_plan = json.load(file)

    return TrainingPlan(training_plan["epochs"],
                        training_plan["batchSize"],
                        training_plan["net"],
                        training_plan["trainingDataQuantity"],
                        training_plan["trainingData"])
