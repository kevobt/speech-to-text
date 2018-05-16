import json
import os
from typing import List

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import (BatchNormalization,
                          Dense,
                          Input,
                          TimeDistributed,
                          Activation,
                          Bidirectional,
                          LSTM,
                          Lambda,
                          GaussianNoise, Conv1D, ZeroPadding1D)

from logger import get_logger
from speech.alphabet import Alphabet
from training.errors.modelnotfound import ModelNotFoundError
from utils.fs import safe_open


class SaveFile:
    def __init__(self, alphabet: Alphabet, model: Model):
        self.alphabet = alphabet
        self.model = model

    def to_json(self):
        return {
            "alphabet": self.alphabet.to_json(),
            "model": self.model.to_json()
        }


def ctc_batch_cost(args):
    """
    Custom implementation of the Keras.backend.ctc_batch_cost function.
    It is needed, because the arguments are mixed up.

    Reference: https://keras.io/backend/

    y_true: tensor (samples, max_string_length) containing the truth labels.
    y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
    input_length: tensor (samples, 1) containing the sequence length for each batch item in y_pred.
    label_length: tensor (samples, 1) containing the sequence length for each batch item in y_true.

    :param args: y_pred, labels, input_length, label_length
    :return: Tensor with shape (samples,1) containing the CTC loss of each element.
    """

    y_pred, y_true, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def add_ctc_loss(model: Model):
    labels = Input(name='labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(model.output_length)(input_length)

    # The lambda layer calls the ctc_batch_cost, which calculates the CTC loss for each element of the batch
    # output_shape=(1,0): corresponds to the CTC loss of the element
    loss_out = Lambda(ctc_batch_cost, output_shape=(1,), name='ctc')(
        [model.output, labels, output_lengths, label_length])

    # Return a new model containing the old one and the new inputs.
    # The output of the model will be the loss calculated by the CTC algorithm.
    return Model(inputs=[model.input, labels, input_length, label_length],
                 outputs=loss_out)


def graves(input_dim=26, rnn_size=512, output_dim=29, std=0.6) -> Model:
    """
    Implementation of the graves model
    Reference: ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    """
    K.set_learning_phase(1)
    input_layer = Input(name='input', shape=(None, input_dim))

    x = BatchNormalization(axis=-1)(input_layer)
    x = GaussianNoise(std)(x)
    x = Bidirectional(LSTM(rnn_size, return_sequences=True))(x)
    x = TimeDistributed(Dense(output_dim))(x)
    prediction_layer = Activation('softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=prediction_layer)
    model.output_length = lambda x: x

    return model


def ds2_gru_model(input_dim=26, fc_size=1024, rnn_size=512, output_dim=29, convolutional_layers=3, lstm_layers=5):
    """
    Reference:
        https://arxiv.org/abs/1512.02595
    """
    pass
    # K.set_learning_phase(1)
    # input_layer = Input(name='input', shape=(None, input_dim))
    #
    # x = BatchNormalization(axis=-1)(input_layer)
    #
    # conv = ZeroPadding1D(padding=(0, 512))(x)
    # for l in range(convolutional_layers):
    #     x = Conv1D(filters=fc_size, name='conv_{}'.format(l + 1), kernel_size=11, padding='valid',
    #                activation='relu', strides=2)(conv)
    #
    # x = BatchNormalization(axis=-1)(x)
    #
    # for l in range(lstm_layers):
    #     x = Bidirectional(LSTM(rnn_size, return_sequences=True))(x)
    #     x = TimeDistributed(Dense(output_dim))(x)
    #
    # x = BatchNormalization(axis=-1)(x)
    # x = TimeDistributed(Dense(output_dim))(x)
    # x = TimeDistributed(Dense(fc_size, activation=clipped_relu))(x)
    # prediction_layer = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(x)
    #
    # model = Model(inputs=input_layer, outputs=prediction_layer)
    # model.output_length = lambda x: x
    #
    # return model


def get(name: str) -> Model:
    if name == 'graves':
        return graves()
    raise ModelNotFoundError(name)


def load(save_file_path: str, weights_path: str) -> SaveFile:
    with open(save_file_path, 'r') as file:
        save_file = json.load(file)

    model = model_from_json(save_file['model'])
    model.load_weights(weights_path)
    model.output_length = lambda x: x
    alphabet = Alphabet(save_file['alphabet'])

    return SaveFile(alphabet, model)


def save(save_file: SaveFile, model_path: str, weights_path: str):
    log = get_logger(__name__)

    # store date to save file
    with safe_open(model_path, 'w') as file:
        file.write(json.dumps(save_file.to_json(), indent=4))
    log.info('Saved model to %s' % os.path.abspath(model_path))

    # save weights
    save_file.model.save_weights(weights_path)
    log.info('Saved weights to %s' % os.path.abspath(weights_path))
