import argparse
import os

from training.trainer import run_training
from training.training import load
from training.trainingstatistics import load as load_statistics
from training.trainingconfig import load as load_config

from logger import get_logger


def main(path: str):
    log = get_logger(name=__name__)

    model_name = os.path.splitext(os.path.basename(path))[0]
    model_base = os.path.join(os.path.dirname(path), model_name)
    config_path = model_base + '.config.json'
    config = load_config(config_path)
    statistics_path = model_base + '.statistics.json'
    statistics = load_statistics(statistics_path)
    weights_path = model_base + '.weights-%s-%s.h5' % (len(statistics.validation_loss), statistics.validation_loss[-1])

    try:
        training = load(path, weights_path, config_path, statistics_path)
        run_training(training, model_base, statistics, config)
    except Exception as ex:
        log.error('%s' % ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue an interrupted training')
    parser.add_argument('path', help='Path to a directory where the training data is stored')
    args = parser.parse_args()

    main(args.path)
