from typing import List


def split(data: List[any], rate=0.5) -> (List[any], List[any]):
    """
    Splits a list into two new ones.
    :param data: list which shall be divided
    :param rate: value between 0 and 1
    :return:
    """
    if not rate >= 0 and rate <= 1:
        raise ValueError('split value is %s but must be between 0 and 1' % split)

    # calculate the list index, where to split the list
    split_index = int(round(len(data) * rate))

    return data[:split_index], data[split_index + 1:]
