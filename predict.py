import argparse

from training.errors.modelnotfound import ModelNotFoundError

from prediction.prediction import predict

from models import load

from logger import get_logger


def main(training_path: str, audio_path: str, weights_path: str):
    log = get_logger(name=__name__)

    save_file = load(training_path, weights_path)

    try:
        log.info('---------------------------------------------')
        log.info('Using net:')
        log.info("{}".format(training_path))
        log.info('---------------------------------------------')
        log.info('Audio path:')
        log.info(audio_path)
        log.info('---------------------------------------------')
        log.info('Predicted transcription:')
        log.info(predict(save_file.model, save_file.alphabet, audio_path))
    except FileNotFoundError as ex:
        log.error('%s' % ex)
    except ModelNotFoundError as ex:
        log.error('%s' % ex)
    except Exception as ex:
        log.error('%s' % ex)


if __name__ == "__main__":
    logger = get_logger(name=__name__)
    parser = argparse.ArgumentParser(description="Predict an audio file transcription")
    parser.add_argument('path', help='Path to a directory where the training data is stored')
    parser.add_argument('weights', help='Path to a weights matrix')
    parser.add_argument('audio', help='Path to audio file which shall be transcribed')
    args = parser.parse_args()

    main(args.path, args.audio, args.weights)
