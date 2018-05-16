import argparse

from training.trainer import run_training
from training.trainingconfig import load as load_training_config
from training.training import create as create_training

from logger import get_logger
from training.trainingstatistics import TrainingStatistics


def main(path: str, training_plan_path: str):
    log = get_logger(__name__)

    config = load_training_config(training_plan_path)

    try:
        training = create_training(training_plan_path)
        run_training(training, path, TrainingStatistics([], []), config)
    except FileExistsError as ex:
        log.error('%s' % ex)
        log.info('Please choose another name for the training. '
                 'You can alternatively overwrite the existing data using the -c flag')
        log.info('Training canceled')
    except Exception as ex:
        log.error('%s' % ex)
        log.info('Training canceled')


if __name__ == '__main__':
    logger = get_logger(name=__name__)

    parser = argparse.ArgumentParser(description='Train a model using the given training plan',)
    parser.add_argument('path', help='Path to new training file')
    parser.add_argument('plan', help='Path to the training configuration')
    args = parser.parse_args()

    main(args.path, args.plan)
