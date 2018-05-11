from typing import List
from keras.optimizers import Adam

from training.callbacks.trainingcallback import TrainingCallback
from training.datagenerator import DataGenerator
from training.trainingdata import TrainingData
from training.training import Training

from models import add_ctc_loss

from logger import get_logger

from utils.list import split


def create_generators(training: Training, rate=0.8) -> (DataGenerator, DataGenerator):
    train_data, validation_data = split(training.training_data, rate)

    return (DataGenerator(train_data, training.alphabet, training.batch_size),
            DataGenerator(validation_data, training.alphabet, training.batch_size))


def calc_steps_per_epoch(data: List[TrainingData], batch_size):
    return len(data) // batch_size


def run_training(training: Training, save_file_path: str):
    log = get_logger(__name__)

    training_data_generator, validation_data_generator = create_generators(training)

    # Take a stochastic gradient descent optimizer
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # add ctc layer to the model
    model = add_ctc_loss(training.model)

    # the CTC algorithm is implemented in teh softmax layer
    # therefore, use a dummy loss function
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    steps_per_epoch = calc_steps_per_epoch(training_data_generator.data, training.batch_size)
    validation_steps = calc_steps_per_epoch(validation_data_generator.data, training.batch_size)

    if steps_per_epoch == 0 or validation_steps == 0:
        raise ValueError("To little data provided. "
                         "Increase the trainingDataQuantity property in the training plan")

    log.info("starting training ...")

    model.fit_generator(generator=training_data_generator.next_batch(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=training.epochs,
                        validation_data=validation_data_generator.next_batch(),
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=[TrainingCallback(training, save_file_path)],
                        initial_epoch=training.passed_epochs)

    log.info('Training finished')
