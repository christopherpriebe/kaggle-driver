# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

import pytest
from kaggle_driver.dataset.data_loc_info import DataLocInfo


def test_initialization_failes_with_missing_mandatory_parameters():
    """Tests that the data location information initialization fails when
    mandatory parameters are missing.
    """
    with pytest.raises(TypeError):
        DataLocInfo("raw/train/path") # pylint: disable=no-value-for-parameter


def test_initialization_with_mandatory_parameters():
    """Tests that the data location information is initialized correctly
    when only mandatory parameters are given.
    """
    data_loc_info = DataLocInfo("raw/train/path", "raw/test/path")
    assert isinstance(data_loc_info, DataLocInfo)
    assert isinstance(data_loc_info.raw_train_dir_path, str)
    assert data_loc_info.raw_train_dir_path == "raw/train/path"
    assert isinstance(data_loc_info.raw_test_dir_path, str)
    assert data_loc_info.raw_test_dir_path == "raw/test/path"
    assert data_loc_info.interim_train_dir_path is None
    assert data_loc_info.interim_test_dir_path is None
    assert data_loc_info.processed_train_dir_path is None
    assert data_loc_info.processed_test_dir_path is None


def test_initialization_with_first_optional_parameter():
    """Tests that the data location information is initialized correctly
    when first optional parameter is given.
    """
    data_loc_info = DataLocInfo(
        "raw/train/path", "raw/test/path",
        interim_train_dir_path="interim/train/path"
    )
    assert isinstance(data_loc_info, DataLocInfo)
    assert isinstance(data_loc_info.raw_train_dir_path, str)
    assert data_loc_info.raw_train_dir_path == "raw/train/path"
    assert isinstance(data_loc_info.raw_test_dir_path, str)
    assert data_loc_info.raw_test_dir_path == "raw/test/path"
    assert isinstance(data_loc_info.interim_train_dir_path, str)
    assert data_loc_info.interim_train_dir_path == "interim/train/path"
    assert data_loc_info.interim_test_dir_path is None
    assert data_loc_info.processed_train_dir_path is None
    assert data_loc_info.processed_test_dir_path is None


def test_initialization_with_second_optional_parameter():
    """Tests that the data location information is initialized correctly
    when second optional parameter is given.
    """
    data_loc_info = DataLocInfo(
        "raw/train/path", "raw/test/path",
        interim_test_dir_path="interim/test/path"
    )
    assert isinstance(data_loc_info, DataLocInfo)
    assert isinstance(data_loc_info.raw_train_dir_path, str)
    assert data_loc_info.raw_train_dir_path == "raw/train/path"
    assert isinstance(data_loc_info.raw_test_dir_path, str)
    assert data_loc_info.raw_test_dir_path == "raw/test/path"
    assert data_loc_info.interim_train_dir_path is None
    assert isinstance(data_loc_info.interim_test_dir_path, str)
    assert data_loc_info.interim_test_dir_path == "interim/test/path"
    assert data_loc_info.processed_train_dir_path is None
    assert data_loc_info.processed_test_dir_path is None


def test_initialization_with_third_optional_parameter():
    """Tests that the data location information is initialized correctly
    when third optional parameter is given.
    """
    data_loc_info = DataLocInfo(
        "raw/train/path", "raw/test/path",
        processed_train_dir_path="processed/train/path"
    )
    assert isinstance(data_loc_info, DataLocInfo)
    assert isinstance(data_loc_info.raw_train_dir_path, str)
    assert data_loc_info.raw_train_dir_path == "raw/train/path"
    assert isinstance(data_loc_info.raw_test_dir_path, str)
    assert data_loc_info.raw_test_dir_path == "raw/test/path"
    assert data_loc_info.interim_train_dir_path is None
    assert data_loc_info.interim_test_dir_path is None
    assert isinstance(data_loc_info.processed_train_dir_path, str)
    assert data_loc_info.processed_train_dir_path == "processed/train/path"
    assert data_loc_info.processed_test_dir_path is None


def test_initialization_with_fourth_optional_parameter():
    """Tests that the data location information is initialized correctly
    when fourth optional parameter is given.
    """
    data_loc_info = DataLocInfo(
        "raw/train/path", "raw/test/path",
        processed_test_dir_path="processed/test/path"
    )
    assert isinstance(data_loc_info, DataLocInfo)
    assert isinstance(data_loc_info.raw_train_dir_path, str)
    assert data_loc_info.raw_train_dir_path == "raw/train/path"
    assert isinstance(data_loc_info.raw_test_dir_path, str)
    assert data_loc_info.raw_test_dir_path == "raw/test/path"
    assert data_loc_info.interim_train_dir_path is None
    assert data_loc_info.interim_test_dir_path is None
    assert data_loc_info.processed_train_dir_path is None
    assert isinstance(data_loc_info.processed_test_dir_path, str)
    assert data_loc_info.processed_test_dir_path == "processed/test/path"


def test_initialization_with_all_parameters():
    """Tests that the data location information is initialized correctly
    when all parameters are given.
    """
    data_loc_info = DataLocInfo(
        "raw/train/path", "raw/test/path",
        "interim/train/path", "interim/test/path",
        "processed/train/path", "processed/test/path"
    )
    assert isinstance(data_loc_info, DataLocInfo)
    assert isinstance(data_loc_info.raw_train_dir_path, str)
    assert data_loc_info.raw_train_dir_path == "raw/train/path"
    assert isinstance(data_loc_info.raw_test_dir_path, str)
    assert data_loc_info.raw_test_dir_path == "raw/test/path"
    assert isinstance(data_loc_info.interim_train_dir_path, str)
    assert data_loc_info.interim_train_dir_path == "interim/train/path"
    assert isinstance(data_loc_info.interim_test_dir_path, str)
    assert data_loc_info.interim_test_dir_path == "interim/test/path"
    assert isinstance(data_loc_info.processed_train_dir_path, str)
    assert data_loc_info.processed_train_dir_path == "processed/train/path"
    assert isinstance(data_loc_info.processed_test_dir_path, str)
    assert data_loc_info.processed_test_dir_path == "processed/test/path"
