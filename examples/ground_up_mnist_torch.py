"""
Ground-Up MNIST Digit Classification with PyTorch
=================================================

An example of a Kaggle competition driver for the MNIST dataset using PyTorch.

This example aims to demonstrate how to use the Kaggle Driver to build an
environment for a Kaggle competition from the ground-up. Therefore, this
example only builds off of the abstract base classes provided by the Kaggle
Driver and does not use any of the provided helper implementations that
are available in the Kaggle Driver.
"""
import shutil
import os
from collections import OrderedDict
from typing import Optional
import numpy.typing as npt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import kaggle_driver as kd


# ==============================================================================
# Define Input and Target subclasses
# ==============================================================================
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


# ==============================================================================
# Define a Dataset subclass using the Input and Target subclasses
# ==============================================================================
class MNISTDataset(kd.Dataset):
    """A class that represents the MNIST dataset.
    """

    def load_train(self) -> OrderedDict[str, tuple[kd.Input, kd.Target]]:
        """Loads the raw training dataset and converts it into an ordered
        dictionary of inputs and targets.

        :return: The ordered dictionary of inputs and targets.
        :rtype: OrderedDict[str, tuple[Input, Target]]
        """
        file_path: str = f"{self.raw_train_dir_path}/train.csv"
        data: npt.NDArray = np.loadtxt(file_path, delimiter=",", skiprows=1)
        inputs: npt.NDArray = data[:, 1:]
        targets: npt.NDArray = data[:, 0]
        train_data: OrderedDict[str, tuple[kd.Input, kd.Target]] = OrderedDict()
        for i, _ in enumerate(inputs):
            train_data[str(i)] = (MNISTImage(inputs[i].reshape(28, 28)),
                                  MNISTLabel(targets[i]))
        return train_data

    def load_test(self) -> OrderedDict[str, kd.Input]:
        """Loads the raw test dataset and converts it into an ordered
        dictionary of inputs.

        :return: The ordered dictionary of inputs.
        :rtype: OrderedDict[str, Input]
        """
        file_path: str = f"{self.raw_test_dir_path}/test.csv"
        data: npt.NDArray = np.loadtxt(file_path, delimiter=",", skiprows=1)
        test_data: OrderedDict[str, kd.Input] = OrderedDict()
        for i, _ in enumerate(data):
            test_data[str(i + 1)] = MNISTImage(data[i].reshape(28, 28))
        return test_data

    def store_predictions(self, predictions_file: str,
                          predictions: dict[str, kd.Target]) -> None:
        """Stores the predictions in a file.

        :param predictions_file: The path to the file to store the predictions
            in.
        :type predictions_file: str
        :param predictions: The predictions to store.
        :type predictions: dict[str, Target]
        """
        with open(predictions_file, "w", encoding="utf-8") as file:
            file.write("ImageId,Label\n")
            for prediction_id, prediction in predictions.items():
                file.write(f"{prediction_id},{prediction.label}\n")


# ==============================================================================
# Define the model, which in this case, is a semi-configurable MLP
# ==============================================================================
@kd.model
class MNISTModel(kd.Model):
    """A configurable MLP model for classification using ReLU
    for all hidden layers and softmax for the output layer.

    :param input_layer_width: The width of the input layer.
    :type input_layer_width: int
    :param output_layer_width: The width of the output layer.
    :type output_layer_width: int
    :param num_hidden_layers: The number of hidden layers.
    :type num_hidden_layers: int
    :param hidden_layer_width: The width of the hidden layers. If
        num_hidden_layers is 0, this parameter is ignored. If num_hidden_layers
        is greater than 0, this parameter must be specified.
    :type hidden_layer_width: Optional[int]
    """
    _input_width: int
    _num_hidden: int
    _hidden_width: int
    _output_width: int
    _model: nn.Sequential

    _DEFAULT_NUM_EPOCHS: int = 5
    _DEFAULT_BATCH_SIZE: int = 64
    _DEFAULT_LEARNING_RATE: float = 0.001
    _DEFAULT_MOMENTUM: float = 0.9

    def __init__(self, input_layer_width: int, output_layer_width: int,
                 num_hidden_layers: int,
                 hidden_layer_width: Optional[int] = None,
                 ) -> None:
        if input_layer_width < 1:
            raise ValueError("input_layer_width must be at least 1")
        if output_layer_width < 1:
            raise ValueError("output_layer_width must be at least 1")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if (num_hidden_layers > 0) and hidden_layer_width is None:
            raise ValueError("hidden_layer_width must be specified if "
                             "num_hidden_layers is non-zero")
        if hidden_layer_width is not None and (hidden_layer_width < 1):
            raise ValueError("hidden_layer_width must be at least 1")

        if num_hidden_layers == 0:
            name: str = f"MNISTModel_{input_layer_width}_{output_layer_width}"
        else:
            name = (f"MNISTModel_{input_layer_width}_{num_hidden_layers}x"
                    f"{hidden_layer_width}_{output_layer_width}")
        super().__init__(name)

        self._input_width = input_layer_width
        self._num_hidden = num_hidden_layers
        self._hidden_width = hidden_layer_width if num_hidden_layers > 0 else 0
        self._output_width = output_layer_width

        model: list[nn.Module] = []
        model.append(nn.Flatten())
        if num_hidden_layers > 0:
            model.append(nn.Linear(input_layer_width, hidden_layer_width))
            model.append(nn.ReLU())
            for _ in range(num_hidden_layers - 1):
                model.append(nn.Linear(hidden_layer_width, hidden_layer_width))
                model.append(nn.ReLU())
            model.append(nn.Linear(hidden_layer_width, output_layer_width))
        else:
            model.append(nn.Linear(input_layer_width, output_layer_width))
        model.append(nn.Softmax(dim=1))
        self._model = nn.Sequential(*model)

    def train(self, train_data: kd.TrainData,
              train_config: kd.TrainConfig) -> kd.TrainResult:
        """Trains the semi-conigurable MLP.

        :param train_data: The training data.
        :type train_data: TrainData
        :param train_config: The training configuration.
        :type train_config: TrainConfig
        :return: The training results.
        :rtype: TrainResult
        """
        def check_data_point(data_point: tuple[kd.Input, kd.Target]) -> None:
            """Checks that the data point is a MNISTImage and MNISTLabel.

            :param data_point: The data point to check.
            :type data_point: tuple[Input, Target]
            :raises TypeError: If the data point is not a MNISTImage and
                MNISTLabel.
            """
            if not isinstance(data_point[0], MNISTImage):
                raise TypeError("data_point[0] must be a MNISTImage")
            if not isinstance(data_point[1], MNISTLabel):
                raise TypeError("data_point[1] must be a MNISTLabel")

        # Create an instance of the default train result class to track
        # statistics during training.
        res = kd.TrainResult()

        # Check that the training data is valid using a local function.
        for data_point in train_data:
            check_data_point(data_point)

        # Check that the training configuration contains a model path which
        # the model will be saved to after training.
        if "model_path" not in train_config:
            raise ValueError("model_path must be specified in train_config")

        # Assign relevant training configuration parameters to local variables.
        # If a parameter is not specified, use the default value.
        model_path: str = train_config["model_path"]
        num_epochs: int = \
            train_config.get("num_epochs", MNISTModel._DEFAULT_NUM_EPOCHS)
        batch_size: int = \
            train_config.get("batch_size", MNISTModel._DEFAULT_BATCH_SIZE)
        learning_rate: float = \
            train_config.get("learning_rate", MNISTModel._DEFAULT_LEARNING_RATE)
        momentum: float = \
            train_config.get("momentum", MNISTModel._DEFAULT_MOMENTUM)

        trainset = torch.utils.data.TensorDataset(
            torch.tensor([d_p[0].image for d_p in train_data.data_points()],
                         dtype=torch.float32),
            torch.tensor([d_p[1].label for d_p in train_data.data_points()],
                         dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(trainset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        # Define the loss function an optimizer using PyTorch for training.
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self._model.parameters(), lr=learning_rate,
                              momentum=momentum)

        # Train the model for the specified number of epochs.
        # Collect the average loss for each epoch and add it to the training
        # results.
        self._model.train()
        res["epoch_average_loss"] = []
        for _ in range(num_epochs):
            epoch_loss: float = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch[0])
                loss = loss_fn(outputs, batch[1])
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            res["epoch_average_loss"].append(epoch_loss / len(dataloader))

        # Save the model to the specified path.
        torch.save(self._model.state_dict(), model_path)

        return res

    def test(self, test_data: kd.TestData, test_config: kd.TestConfig) \
        -> tuple[dict[str, kd.Target], kd.TestResult]:
        """Tests the semi-configurable MLP.

        :param test_data: The testing data.
        :type test_data: dict[str, Input]
        :param test_config: The test configuration.
        :type test_config: TestConfig
        :return: The predictions and the testing results.
        :rtype: tuple[dict[str, Target], TestResult]
        """
        def check_data_point(data_point: kd.Input) -> None:
            """Checks that the data point is a MNISTImage.

            :param data_point: The data point to check.
            :type data point: Input
            :raises TypeError: If the data point is not a MNISTImage.
            """
            if not isinstance(data_point, MNISTImage):
                raise TypeError("data_point must be a MNISTImage")

        # Create an instance of the default test result class to track
        # statistics during testing.
        res = kd.TestResult()

        # Check that the testing data is valid using a local function.
        for data_point in test_data:
            check_data_point(data_point)

        # Check that the training configuration contains a model path which
        # the model will be saved to after training.
        if "model_path" not in test_config:
            raise ValueError("model_path must be specified in test_config")

        # Assign relevant training configuration parameters to local variables.
        # If a parameter is not specified, use the default value.
        model_path: str = test_config["model_path"]

        # Load the model from the specified path.
        self._model.load_state_dict(torch.load(model_path))

        # Test the model by generating predictions given the test data.
        self._model.eval()
        predictions: dict[str, kd.Target] = {}
        with torch.no_grad():
            for data_point_id, data_point in zip(test_data.data_point_ids(),
                                                test_data.data_points()):
                inputs: torch.Tensor = torch.tensor(data_point.image,
                                                    dtype=torch.float32)
                outputs: torch.Tensor = self._model(inputs.unsqueeze(0))
                raw_prediction: int = torch.argmax(outputs).item()
                predictions[data_point_id] = MNISTLabel(raw_prediction)

        return predictions, res


# ==============================================================================
# Set the name of the competition on Kaggle
# ==============================================================================
competition_name: str = "digit-recognizer"


# ==============================================================================
# Define the location of the data
# ==============================================================================
curr_dir: str = os.path.dirname(os.path.abspath(__file__))
data_loc_info = kd.DataLocInfo(
    raw_train_dir_path=curr_dir + "/../data/raw/train/mnist",
    raw_test_dir_path=curr_dir + "/../data/raw/test/mnist",
    interim_train_dir_path=curr_dir + "/../data/interim/train/mnist",
    interim_test_dir_path=curr_dir + "/../data/interim/test/mnist",
    processed_train_dir_path=curr_dir + "/..data/processed/train/mnist",
    processed_test_dir_path=curr_dir + "/..data/processed/test/mnist",
)


# ==============================================================================
# Define a function to organize the downloaded data into the correct folders
# ==============================================================================
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


# ==============================================================================
# Set information to be used when interfacing with Kaggle directly
# NOTE: This is optional and only needed if you want to use the Kaggle API
#       Alternatively, you can download the data manually and organize it
#       along with uploading the submission file manually
# ==============================================================================
kaggle_info = kd.KaggleInfo(
    competition_name=competition_name,
    organize_data_fn=organize_mnist_data,
)


# ==============================================================================
# Initialize the dataset and driver and call the driver CLI
# ==============================================================================
dataset = MNISTDataset(data_loc_info)
if __name__ == "__main__":
    kd.run(dataset, kaggle_info=kaggle_info)
