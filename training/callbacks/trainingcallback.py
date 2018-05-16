import os

from keras.callbacks import Callback

from training.training import Training

from models import save as save_model, SaveFile

from training.trainingstatistics import save as save_statistics, TrainingStatistics
from training.trainingconfig import save as save_config, TrainingConfig


class TrainingCallback(Callback):
    def __init__(self, path: str, statistics: TrainingStatistics, training: Training, config: TrainingConfig):
        super().__init__()
        self.path = path
        self.statistics = statistics
        self.training = training
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        validation_loss = logs['val_loss']

        self.statistics.loss.append(loss)
        self.statistics.validation_loss.append(validation_loss)

        save_model(SaveFile(self.training.alphabet, self.training.model),
                   self.path + '.json',
                   self.path + '.weights-%s-%s.h5' % (epoch + 1, validation_loss))
        save_statistics(self.path + '.statistics.json', self.statistics)
        save_config(self.path + '.config.json', self.config)

