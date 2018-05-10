from typing import List
from speech.processing import calculate_mfccs
from rx import Observable

import numpy as np

from speech.alphabet import Alphabet, text_to_int_sequence

from training.trainingdata import TrainingData


class Batch:
    def __init__(self, data: List[TrainingData], alphabet: Alphabet):
        self.data = data
        self.alphabet = alphabet
        self.max_length = 0
        self.max_string_length = 0
        self.features = []

        # Calculate mfccs
        Observable.from_(self.data) \
            .map(lambda item: item.path) \
            .map(lambda path: calculate_mfccs(path)) \
            .subscribe(lambda features: self.add_features(features))

        # Determine longest mfcc
        Observable.from_(self.features) \
            .map(lambda mfcc: mfcc.shape[0]) \
            .max() \
            .subscribe(lambda max_length: self.set_max_length(max_length))

        # Determine longest text
        Observable.from_(self.data) \
            .map(lambda item: item.text) \
            .map(lambda text: len(text)) \
            .max() \
            .subscribe(lambda max_string_length: self.set_max_string(max_string_length))

    def add_features(self, features):
        self.features.append(features)

    def set_max_length(self, length):
        self.max_length = length

    def set_max_string(self, length):
        self.max_string_length = length

    def get_data(self):
        return self.data

    def get(self):
        # Create Matrices which are going to hold the input and the output of the batch items
        input = np.zeros([len(self.data), self.max_length, 26])
        labels = np.ones([len(self.data), self.max_string_length]) * len(self.alphabet)

        # Needed for CTC loss algorithm
        input_length = np.zeros([len(self.data), 1])
        label_length = np.zeros([len(self.data), 1])

        for i in range(0, len(self.data)):
            # Insert the features of each batch item into the created input matrix
            feat = self.features[i]
            input_length[i] = feat.shape[0]
            input[i, :feat.shape[0], :] = feat

            # the labels will be represented as numbers. They will be added into the created label matrix
            label = np.array(text_to_int_sequence(self.data[i].text, self.alphabet))
            label_length[i] = len(label)
            labels[i, :len(label)] = label

        # The output will be the Lambda layer called 'ctc'.
        # A matrix will hold the predicted labels
        outputs = {'ctc': np.zeros([len(self.data)])}

        # The final input lists all parameters needed for the CTC algorithm
        inputs = {
            'input': input,
            'labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }

        return inputs, outputs


class DataGenerator:
    def __init__(self,
                 data: List[TrainingData],
                 alphabet: Alphabet,
                 batch_size=20):

        self.data = data
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.current_batch_index = 0

    def get_batch(self):
        return Batch(self.data[self.current_batch_index:self.current_batch_index + self.batch_size], self.alphabet)\
            .get()

    def next_batch(self) -> Batch:
        while True:
            ret = self.get_batch()
            self.current_batch_index += self.batch_size

            if self.current_batch_index >= len(self.data) - self.batch_size:
                self.current_batch_index = 0
                self.shuffle_data()

            yield ret

    def shuffle_data(self):
        # One could use the validation_split and shuffle option when training, but the validation
        # data won't be shuffled. See issue https://github.com/keras-team/keras/issues/597 or
        # http://forums.fast.ai/t/for-keras-fit-method-does-shuffle-true-shuffle-both-the-training-and-validation-samples-or-just-the-training-dataset/2992
        p = np.random.permutation(len(self.data))
        self.data = [self.data[i] for i in p]
