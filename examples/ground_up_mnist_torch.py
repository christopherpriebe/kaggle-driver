"""Ground-up MNIST digit classification with PyTorch.

Demonstrates how to use kaggle-driver from first principles, without any of
the optional integrations helpers. The user supplies:

* ``MNISTDataset`` (a :class:`kaggle_driver.Dataset` subclass) that loads the
  Kaggle ``digit-recognizer`` train/test CSVs.
* ``MNISTModel`` (a :class:`kaggle_driver.Model` subclass) that owns a
  configurable multilayer perceptron and implements train/test/save/load with raw PyTorch.

The driver script wires them together with :func:`kaggle_driver.run`.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import kaggle_driver as kd

FloatArray = npt.NDArray[np.float32]
COMPETITION_NAME = "digit-recognizer"


class MNISTDataset(kd.Dataset[FloatArray, int]):
    """Dataset over the Kaggle digit-recognizer CSV files."""

    def load_train(self) -> Mapping[str, tuple[FloatArray, int]]:
        """Load ``train.csv`` into a mapping of example id to (image, label)."""
        rows = np.loadtxt(self.raw_train_directory / "train.csv", delimiter=",", skiprows=1)
        labels = rows[:, 0].astype(np.int64)
        images = rows[:, 1:].astype(np.float32).reshape(-1, 28, 28)
        return {str(i): (images[i], int(labels[i])) for i in range(len(rows))}

    def load_test(self) -> Mapping[str, FloatArray]:
        """Load ``test.csv`` into a mapping of example id to image."""
        rows = np.loadtxt(self.raw_test_directory / "test.csv", delimiter=",", skiprows=1)
        images = rows.astype(np.float32).reshape(-1, 28, 28)
        return {str(i + 1): images[i] for i in range(len(rows))}

    def store_predictions(
        self,
        path: Path,
        predictions: Mapping[str, int],
    ) -> None:
        """Write a Kaggle-style ``ImageId,Label`` submission CSV."""
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ImageId,Label\n")
            for image_id, label in predictions.items():
                handle.write(f"{image_id},{label}\n")


class MNISTModel(kd.Model[FloatArray, int]):
    """Configurable multilayer perceptron with ReLU hidden layers and a softmax output.

    Args:
        input_layer_width: Number of input features (typically ``784``).
        output_layer_width: Number of output classes (``10`` for MNIST).
        hidden_layer_count: Number of hidden layers; zero produces a
            single linear layer.
        hidden_layer_width: Width of each hidden layer; required if
            ``hidden_layer_count > 0``.
    """

    def __init__(
        self,
        input_layer_width: int,
        output_layer_width: int,
        hidden_layer_count: int,
        hidden_layer_width: int | None = None,
    ) -> None:
        if input_layer_width < 1 or output_layer_width < 1:
            raise ValueError("input_layer_width and output_layer_width must be >= 1")
        if hidden_layer_count < 0:
            raise ValueError("hidden_layer_count must be non-negative")
        if hidden_layer_count > 0 and hidden_layer_width is None:
            raise ValueError("hidden_layer_width is required when hidden_layer_count > 0")
        if hidden_layer_width is not None and hidden_layer_width < 1:
            raise ValueError("hidden_layer_width must be >= 1")

        self._init_kwargs = {
            "input_layer_width": input_layer_width,
            "output_layer_width": output_layer_width,
            "hidden_layer_count": hidden_layer_count,
            "hidden_layer_width": hidden_layer_width,
        }

        layers: list[nn.Module] = [nn.Flatten()]
        if hidden_layer_count == 0:
            layers.append(nn.Linear(input_layer_width, output_layer_width))
        else:
            assert hidden_layer_width is not None  # narrowed by the checks above
            layers.append(nn.Linear(input_layer_width, hidden_layer_width))
            layers.append(nn.ReLU())
            for _ in range(hidden_layer_count - 1):
                layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layer_width, output_layer_width))
        layers.append(nn.Softmax(dim=1))
        self._module = nn.Sequential(*layers)

    def train(
        self,
        train_data: Mapping[str, tuple[FloatArray, int]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Fit the multilayer perceptron and save it to ``config['model_path']``."""
        if "model_path" not in config:
            raise ValueError("train config must include 'model_path'")
        epoch_count = int(config.get("epoch_count", 5))
        batch_size = int(config.get("batch_size", 64))
        learning_rate = float(config.get("learning_rate", 0.001))
        momentum = float(config.get("momentum", 0.9))

        inputs = torch.tensor(np.stack([pair[0] for pair in train_data.values()]))
        targets = torch.tensor([pair[1] for pair in train_data.values()], dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self._module.parameters(), lr=learning_rate, momentum=momentum)

        self._module.train()
        epoch_losses: list[float] = []
        for _ in range(epoch_count):
            running_loss = 0.0
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                outputs = self._module(batch_inputs)
                loss = loss_function(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            epoch_losses.append(running_loss / len(loader))

        self.save(Path(config["model_path"]))
        return {"epoch_average_loss": epoch_losses}

    def test(
        self,
        test_data: Mapping[str, FloatArray],
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, int], Mapping[str, Any]]:
        """Load weights from ``config['model_path']`` and predict each test row."""
        if "model_path" not in config:
            raise ValueError("test config must include 'model_path'")
        checkpoint = torch.load(Path(config["model_path"]), weights_only=False)
        self._module.load_state_dict(checkpoint["state_dict"])

        self._module.eval()
        predictions: dict[str, int] = {}
        with torch.no_grad():
            for example_id, image in test_data.items():
                tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                predictions[example_id] = int(torch.argmax(self._module(tensor)).item())
        return predictions, {}

    def save(self, path: Path) -> None:
        """Persist constructor kwargs and weights to a single file."""
        torch.save(
            {"init_kwargs": self._init_kwargs, "state_dict": self._module.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MNISTModel:
        """Reconstruct a model previously written by :meth:`save`."""
        checkpoint = torch.load(path, weights_only=False)
        instance = cls(**checkpoint["init_kwargs"])
        instance._module.load_state_dict(checkpoint["state_dict"])
        return instance


def organize_mnist_data(
    unzipped_directory: Path,
    raw_train_directory: Path,
    raw_test_directory: Path,
) -> None:
    """Move ``train.csv`` and ``test.csv`` into the dataset's raw directories."""
    shutil.move(unzipped_directory / "train.csv", raw_train_directory / "train.csv")
    shutil.move(unzipped_directory / "test.csv", raw_test_directory / "test.csv")


_HERE = Path(__file__).resolve().parent
dataset = MNISTDataset(
    raw_train_directory=_HERE / "data" / "raw" / "train" / "mnist",
    raw_test_directory=_HERE / "data" / "raw" / "test" / "mnist",
)
kaggle_info = kd.KaggleInfo(
    competition_name=COMPETITION_NAME,
    organize_data_function=organize_mnist_data,
)


if __name__ == "__main__":
    kd.run(dataset, models={"multilayer_perceptron": MNISTModel}, kaggle_info=kaggle_info)
