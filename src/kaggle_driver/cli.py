"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m kaggle_driver` python will execute
    ``__main__.py`` as a script. That means there will not be any
    ``kaggle_driver.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there"s no ``kaggle_driver.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import argparse
from typing import Optional
from kaggle_driver.dataset import Dataset
from kaggle_driver.directory.model_directory import ModelDirectory
from kaggle_driver.driver.driver import Driver
from kaggle_driver.driver.kaggle_info import KaggleInfo


def _builtin_main() -> None:
    """Entry point for the application script.
    """
    raise NotImplementedError("This entry point for Kaggle Driver is not \
                            implemented yet")


def run(dataset: Dataset, kaggle_info: Optional[KaggleInfo] = None) -> None:
    """Entry point for the application script built by the user.

    :param dataset: The dataset.
    :type dataset: Dataset
    :param kaggle_info: The information about the Kaggle competition. If None,
        then the dataset is assumed to be already downloaded and organized
        into the correct folders.
    :type kaggle_info: Optional[KaggleInfo]
    """
    _user_main(dataset, kaggle_info)


def _user_main(dataset: Dataset, kaggle_info: Optional[KaggleInfo]) -> None:
    models: list[str] = ModelDirectory.keys()

    description: str = "A driver for Kaggle competitions."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output to stdout.")
    # TODO: Add an argument for logging

    subparsers = parser.add_subparsers(dest="subcommands")

    download_parser = subparsers.add_parser("download",
                                            help="Download the dataset.")

    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("model", choices=models,
                              help="The model to train.")
    train_parser.add_argument("--model_config_file", type=str,
                              required=False,
                              help="The path to the model config file.")
    train_parser.add_argument("--train_config_file", type=str,
                              required=False,
                              help="The path to the training config file.")

    test_parser = subparsers.add_parser("test", help="Test a model.")
    test_parser.add_argument("model", choices=models,
                             help="The model to test.")
    test_parser.add_argument("--submission_file", type=str,
                             help="The path to the submission file.")
    test_parser.add_argument("--model_config_file", type=str,
                             required=False,
                             help="The path to the model config file.")
    test_parser.add_argument("--test_config_file", type=str,
                             required=False,
                             help="The path to the testing config file.")

    args: argparse.Namespace = parser.parse_args()
    driver = Driver(dataset, kaggle_info=kaggle_info, verbose=args.verbose)

    if args.subcommands == "download":
        driver.download()
    elif args.subcommands == "train":
        driver.train(args.model, args.model_config_file,
                     args.train_config_file)
    elif args.subcommands == "test":
        driver.test(args.model, args.model_config_file, args.test_config_file,
                    args.submission_file)
    else:
        raise ValueError("Invalid subcommand.")
