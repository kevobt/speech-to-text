import os
import json
import logging.config

from utils.fs import safe_open

def setup_logging(default_path='./logging.json', default_level=logging.INFO):
    path = default_path

    if os.path.exists(path):
        with safe_open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def get_logger(name=None, default_path='./logging.json'):
    setup_logging(default_path)
    return logging.getLogger(name)
