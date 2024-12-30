"""
General utils
"""

import os
import json


def create_directory(path):
    """
    create directory
    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def save_json(dictionery, json_path):
    with open(json_path, "w") as outfile:
        json.dump(dictionery, outfile)


def load_json(json_path):
    with open(json_path, "r") as file:
        obj = json.load(file)
    return obj
