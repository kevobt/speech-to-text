import os


def safe_open(filename: str, mode: str):
    """
    Writes to a file. In addition, it creates the directories, if necessary
    :param filename: path to the file
    :param mode: writing mode
    :return:
    """
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as ex:
        pass

    return open(filename, mode)
