"""A module that contains the driver class.
"""
import os
import shutil
import zipfile
from typing import Any, Optional
from kaggle_driver.dataset import Dataset
from kaggle_driver.directory import ModelDirectory
from kaggle_driver.model import Model, TestResult, TrainResult
from kaggle_driver.pprint import pprint_dict
from kaggle_driver.utils import make_dir_if_not_exists, yaml_to_dict
from kaggle_driver.verbose_print import print_msg, print_title
from .kaggle_info import KaggleInfo


class Driver:
    """The driver class that orchestrates execution.

    :param dataset: The dataset.
    :type dataset: Dataset
    :param kaggle_info: The information about the Kaggle competition. If None,
        then the dataset is assumed to be already downloaded and organized
        into the correct folders.
    :type kaggle_info: Optional[KaggleInfo]
    :param verbose: Whether to print verbose output to stdout.
    :type verbose: bool
    """
    _YAML_FILE_EXTENSIONS: tuple[str, ...] = (".yaml", ".yml")
    _dataset: Dataset
    _kaggle_info: Optional[KaggleInfo]
    _verbose: bool

    def __init__(self, dataset: Dataset,
                 kaggle_info: Optional[KaggleInfo] = None,
                 verbose: bool = False) -> None:
        self._dataset = dataset
        self._kaggle_info = kaggle_info
        self._verbose = verbose


    def download(self) -> None:
        """Download the dataset from Kaggle.
        """
        if self._kaggle_info is None:
            raise ValueError("Kaggle info must be provided to download "
                                "the dataset.")
        self._download_dataset(self._kaggle_info)

    def _download_dataset(self, kaggle_info: KaggleInfo) -> None:
        import kaggle  # pylint: disable=import-outside-toplevel
        from kaggle import api  # pylint: disable=import-outside-toplevel

        def authenticate(_api: kaggle.KaggleApi) -> None:
            _api.authenticate()

        def download_competition_files(_api: kaggle.KaggleApi,
                                       competition_name: str,
                                       path: str) -> None:
            _api.competition_download_files(competition_name, path=path)

        authenticate(api)
        temp_data_dir_path = f"{os.getcwd()}/.tmp"
        make_dir_if_not_exists(temp_data_dir_path)
        download_competition_files(api, kaggle_info.competition_name,
                                      temp_data_dir_path)
        zip_file_name: str = os.listdir(temp_data_dir_path)[0]
        with zipfile.ZipFile(f"{temp_data_dir_path}/{zip_file_name}", "r") \
                as zip_file:
            zip_file.extractall(temp_data_dir_path)

        try:
            kaggle_info.organize_data_fn(temp_data_dir_path,
                                        self._dataset.raw_train_dir_path,
                                        self._dataset.raw_test_dir_path)
        except Exception as exception:  # pylint: disable=broad-except
            shutil.rmtree(temp_data_dir_path)
            raise exception

        shutil.rmtree(temp_data_dir_path)

    def train(self, model_name: str, model_config_file: Optional[str],
              train_config_file: Optional[str]) -> None:
        """Trains a model.

        :param model_name: The name of the model.
        :type model_name: str
        :param model_config_file: The path to the model configuration file.
        :type model_config_file: Optional[str]
        :param train_config_file: The path to the train configuration file.
        :type train_config_file: Optional[str]
        """
        model_config: dict[str, Any] = self.read_configuration_file(
            model_config_file) if model_config_file is not None else {}
        train_config: dict[str, Any] = self.read_configuration_file(
            train_config_file) if train_config_file is not None else {}
        model_type: type = ModelDirectory.get(model_name)
        model: Model = model_type(**model_config)

        if self._verbose:
            print_title("Training")
            print_msg(f"Model configuration:\n{pprint_dict(model_config)}")
            print_msg(f"Train configuration:\n{pprint_dict(train_config)}")
            print_msg(f"Model: {model.name}")

        train_results: TrainResult = model.train(self._dataset.train,
                                                 train_config)

        if self._verbose:
            print_msg(f"Train results:\n{pprint_dict(train_results)}")

    def test(self, model_name: str, model_config_file: Optional[str],
             test_config_file: Optional[str], submission_file: str) -> None:
        """Tests a model.

        :param model_name: The name of the model.
        :type model_name: str
        :param model_config_file: The path to the model configuration file.
        :type model_config_file: Optional[str]
        :param test_config_file: The path to the test configuration file.
        :type test_config_file: Optional[str]
        :param submission_file: The path to the submission file.
        :type submission_file: str
        """
        model_config: dict[str, Any] = self.read_configuration_file(
            model_config_file) if model_config_file is not None else {}
        test_config: dict[str, Any] = self.read_configuration_file(
            test_config_file) if test_config_file is not None else {}
        model_type: type = ModelDirectory.get(model_name)
        model: Model = model_type(**model_config)

        if self._verbose:
            print_title("Testing")
            print_msg(f"Model configuration:\n{pprint_dict(model_config)}")
            print_msg(f"Test configuration:\n{pprint_dict(test_config)}")
            print_msg(f"Model: {model.name}")

        predictions, test_results = model.test(self._dataset.test, test_config)
        self._dataset.store_predictions(submission_file, predictions)

        if self._verbose:
            print_msg(f"Test results:\n{pprint_dict(test_results)}")
            print_msg(f"Stored predictions in {submission_file}.")

    @staticmethod
    def read_configuration_file(file_path: str) -> dict[str, Any]:
        """Reads a configuration file in varying formats.
        Supported formats: YAML

        :param file_path: The path to the configuration file.
        :type file_path: str
        :return: The configuration.
        :rtype: dict[str, Any]
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Configuration file {file_path} does not exist.")

        if file_path.endswith(Driver._YAML_FILE_EXTENSIONS):
            return yaml_to_dict(file_path)
        else:
            raise ValueError(f"Configuration file {file_path} has an "
                                "unsupported file extension.")
