"""An example of a Kaggle competition driver for the MNIST dataset.
"""
import kaggle_driver as kd
import numpy as np
import numpy.typing as npt
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim


class MNISTImage(kd.Input):
    """A class that represents an image in the MNIST dataset.

    :param image: The image.
    :type image: npt.NDArray
    """
    _image: npt.NDArray

    def __init__(self, image: npt.NDArray) -> None:
        super().__init__()
        self._image = image

    @property
    def image(self) -> npt.NDArray:
        """The image.

        :return: The image.
        :rtype: npt.NDArray
        """
        return self._image


class MNISTLabel(kd.Target):
    """A class that represents a label in the MNIST dataset.

    :param label: The label.
    :type label: int
    """
    _label: int

    def __init__(self, label: int) -> None:
        super().__init__()
        self._label = label

    @property
    def label(self) -> int:
        """The label.

        :return: The label.
        :rtype: int
        """
        return self._label


class MNISTDataset(kd.Dataset):
    """A class that represents the MNIST dataset.
    """
    def load_train(self) -> dict[str, tuple[kd.Input, kd.Target]]:
        """Loads the raw training dataset and converts it into a dictionary
            of inputs and targets.

        :return: The dictionary of inputs and targets.
        :rtype: dict[str, tuple[Input, Target]]
        """
        file_path: str = f"{self.raw_train_dir_path}/train.csv"
        data: npt.NDArray = np.loadtxt(file_path, delimiter=",", skiprows=1)
        inputs: npt.NDArray = data[:, 1:]
        targets: npt.NDArray = data[:, 0]
        train_data: dict[str, tuple[kd.Input, kd.Target]] = {}
        for i, _ in enumerate(inputs):
            train_data[str(i)] = (MNISTImage(inputs[i].reshape(28, 28)),
                                  MNISTLabel(targets[i]))
        return train_data

    def load_test(self) -> dict[str, kd.Input]:
        """Loads the raw test dataset and converts it into a dictionary of
            inputs.

        :return: The dictionary of inputs.
        :rtype: dict[str, Input]
        """
        file_path: str = f"{self.raw_test_dir_path}/test.csv"
        data: npt.NDArray = np.loadtxt(file_path, delimiter=",", skiprows=1)
        test_data: dict[str, kd.Input] = {}
        for i, _ in enumerate(data):
            test_data[str(i)] = MNISTImage(data[i].reshape(28, 28))
        return test_data


# @kd.model
# class MNISTModel(kd.Model):
#     pass


competition_name: str = "digit-recognizer"

data_loc_info = kd.DataLocInfo(
    raw_train_dir_path="data/raw/train/mnist",
    raw_test_dir_path="data/raw/test/mnist",
    interim_train_dir_path="data/interim/train/mnist",
    interim_test_dir_path="data/interim/test/mnist",
    processed_train_dir_path="data/processed/train/mnist",
    processed_test_dir_path="data/processed/test/mnist",
)

def organize_mnist_data(unzipped_data_dir: str,
                        raw_train_data_dir: str,
                        raw_test_data_dir: str) -> None:
    """Organizes the MNIST data.

    :param unzipped_data_dir: The directory containing the unzipped data.
    :type unzipped_data_dir: str
    :param raw_train_data_dir: The directory where the raw training data should
        be stored.
    :type raw_train_data_dir: str
    :param raw_test_data_dir: The directory where the raw test data should be
        stored.
    :type raw_test_data_dir: str
    """
    if not os.path.exists(raw_train_data_dir):
        os.makedirs(raw_train_data_dir)
    if not os.path.exists(raw_test_data_dir):
        os.makedirs(raw_test_data_dir)
    train_data_file_suffix: str = "train.csv"
    test_data_file_suffix: str = "test.csv"
    shutil.move(f"{unzipped_data_dir}/{train_data_file_suffix}",
                f"{raw_train_data_dir}/{train_data_file_suffix}")
    shutil.move(f"{unzipped_data_dir}/{test_data_file_suffix}",
                f"{raw_test_data_dir}/{test_data_file_suffix}")

kaggle_info = kd.KaggleInfo(
    competition_name=competition_name,
    organize_data_fn=organize_mnist_data,
)

dataset = MNISTDataset(data_loc_info)
driver = kd.Driver(dataset, kaggle_info)
