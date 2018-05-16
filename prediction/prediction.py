import os
import numpy as np
import keras.backend as K
from keras import Model

from speech.processing import calculate_mfccs
from speech.alphabet import int_to_text_sequence, Alphabet

from training.training import Training


def predict(model: Model, alphabet: Alphabet, audio_path: str) -> str:
    """
    Creates a transcription of an audio file using a trained model
    :param model: The model to use
    :param alphabet: Alphabet containing the possible characters of the transcription
    :param audio_path: path to the audio file to transcribe
    :return: transcription
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError('The audio file "%s" does not exist' % audio_path)

    mfccs = calculate_mfccs(audio_path)

    # the predict method expects a numpy array of inputs to be predicted. Since there is only one element to predict,
    # the input are the mfccs of one audio file. The input matrix is created by expanding the mfccs matrix of shape
    # (number of frames, 26) by one axis resulting in (1, number of frames, 26).
    prediction = model.predict(np.expand_dims(mfccs, axis=0))

    # The ctc_decode method expects a list containing the length of each element to be predicted.
    # Since there is only element, the list contains the elements length (number of frames).
    input_length = [mfccs.shape[0]]

    # ctc_decode returns a Tuple. the first element is a list containing the decoded sequence.
    # Reference: https://keras.io/backend/
    decoded_sequence_tensor = K.ctc_decode(prediction, input_length)[0][0]

    # The decoded sequence itself is a tensor and must therefore be evaluated using K.eval first
    decoded_sequence = K.eval(decoded_sequence_tensor)

    # The array decoded sequence must be flattened and converted from a numpy.ndarray into an ordinary array.
    predicted_ints = decoded_sequence.flatten().tolist()

    return ''.join(int_to_text_sequence(predicted_ints, alphabet))
