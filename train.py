import argparse

from training.trainer import run_training
from training.trainingconfig import load as load_training_config
from training.training import create as create_training

from logger import get_logger


def main(path: str, training_plan_path: str, overwrite: bool):
    log = get_logger(__name__)

    try:
        training = create_training(path, load_training_config(training_plan_path), overwrite)
        run_training(training, path)
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
    parser.add_argument('-c', '--overwrite', action='store_true', help='Replaces other training data if necessary')
    args = parser.parse_args()

    main(args.path, args.plan, args.overwrite)
