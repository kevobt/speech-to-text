import os
import json

from typing import List


class Alphabet:
    def __init__(self, characters: List[str]):
        self.characters = [" "]
        self.characters.extend(characters)

    def __len__(self):
        return len(self.characters)

    def __contains__(self, item):
        return item in self.characters

    def to_json(self):
        return self.characters


def load(path: str) -> Alphabet:
    if not os.path.exists(path):
        raise FileNotFoundError('The path "%s" to the alphabet does not exist' % path)

    with open(path, 'r') as file:
        alphabet_json = json.load(file)

    return Alphabet(alphabet_json)


def text_to_int_sequence(text: str, alphabet: Alphabet) -> List[int]:
    """
    Covnerts a text into a sequence of ints
    :param text: text to convert
    :param alphabet: alphabet
    :return: Int sequence
    """
    return [alphabet.characters.index(character) for character in text]


def int_to_text_sequence(text: List[int], alphabet: Alphabet) -> List[str]:
    """
    Covnerts a sequence of ints into a text
    :param text: ints to convert
    :param alphabet: alphabet
    :return: text
    """
    return [alphabet.characters[character] for character in text]