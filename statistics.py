import argparse
import matplotlib.pyplot as plot

from training.trainingstatistics import load

from logger import get_logger


def main(path: str):
    log = get_logger(__name__)

    try:
        training = load(path)

        # plot loss and validation loss
        plot.plot(training.loss)
        plot.plot(training.validation_loss)
        plot.title('model loss')
        plot.ylabel('loss')
        plot.xlabel('epoch')
        plot.legend(['train', 'validation'], loc='upper left')
        plot.show()
    except FileNotFoundError as ex:
        log.error('%s' % ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for displaying statistics of trainings")
    parser.add_argument("path", help="Path to a training statistics file")
    args = parser.parse_args()

    main(args.path)
