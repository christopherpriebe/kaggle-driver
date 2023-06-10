"""Utility functions for the kaggle_driver package.
"""
import os


def make_dir_if_not_exists(path: str) -> None:
    """Makes a directory if it does not exist.

    :param path: The path to the directory.
    :type path: str
    """
    if not os.path.exists(path):
        os.mkdir(path)
