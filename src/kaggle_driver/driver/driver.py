"""A module that contains the driver class.
"""
import os
import shutil
import zipfile
from typing import Optional
from kaggle_driver.dataset import Dataset
from kaggle_driver.utils import make_dir_if_not_exists
from .kaggle_info import KaggleInfo


class Driver:
    """The driver class that orchestrates execution.

    :param dataset: The dataset.
    :type dataset: Dataset
    :param kaggle_info: The information about the Kaggle competition. If None,
        then the dataset is assumed to be already downloaded and organized
        into the correct folders.
    :type kaggle_info: Optional[KaggleInfo]
    :default kaggle_info: None
    """
    _dataset: Dataset

    def __init__(self, dataset: Dataset,
                 kaggle_info: Optional[KaggleInfo] = None) -> None:
        self._dataset = dataset
        if kaggle_info is not None:
            self._download_dataset(kaggle_info)

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
