import argparse

from training.trainer import run_training
from training.training import load

from logger import get_logger


def main(path: str):
    log = get_logger(name=__name__)

    try:
        training = load(path)
        run_training(training, path)
    except Exception as ex:
        log.error('%s' % ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue an interrupted training')
    parser.add_argument('path', help='Path to a directory where the training data is stored')
    args = parser.parse_args()

    main(args.path)
