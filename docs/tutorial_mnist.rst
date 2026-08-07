==================================================
Tutorial: ground-up MNIST classification (PyTorch)
==================================================

This tutorial walks through the
``examples/ground_up_mnist_torch.py`` example, which builds a PyTorch
pipeline on the Kaggle ``digit-recognizer`` competition without using
:class:`~kaggle_driver.integrations.torch.TorchModel`. It shows what the
framework asks of you when you subclass the bare
:class:`~kaggle_driver.Model` ABC directly.

Install the extras
==================

.. code-block:: bash

    pip install "kaggle-driver[torch,kaggle]"

Subclass ``Dataset`` for MNIST
==============================

MNIST CSVs are flat: each row is one image plus its label. We parse them
into NumPy arrays and pair each image with its label.

.. code-block:: python

    class MNISTDataset(kd.Dataset[FloatArray, int]):
        def load_train(self) -> Mapping[str, tuple[FloatArray, int]]:
            rows = np.loadtxt(self.raw_train_directory / "train.csv", delimiter=",", skiprows=1)
            labels = rows[:, 0].astype(np.int64)
            images = rows[:, 1:].astype(np.float32).reshape(-1, 28, 28)
            return {str(i): (images[i], int(labels[i])) for i in range(len(rows))}

        def load_test(self) -> Mapping[str, FloatArray]:
            rows = np.loadtxt(self.raw_test_directory / "test.csv", delimiter=",", skiprows=1)
            images = rows.astype(np.float32).reshape(-1, 28, 28)
            return {str(i + 1): images[i] for i in range(len(rows))}

        def store_predictions(self, path, predictions):
            with path.open("w", encoding="utf-8") as handle:
                handle.write("ImageId,Label\n")
                for image_id, label in predictions.items():
                    handle.write(f"{image_id},{label}\n")

Subclass ``Model`` for a configurable multilayer perceptron
===========================================================

The model owns its architecture and implements ``train``, ``test``,
``save``, and ``load``. The constructor takes the architecture
hyperparameters (which the framework forwards from the model config
YAML); ``train`` and ``test`` receive their own per-call configs.

.. code-block:: python

    class MNISTModel(kd.Model[FloatArray, int]):
        def __init__(self, input_layer_width, output_layer_width,
                     hidden_layer_count, hidden_layer_width=None):
            self._init_kwargs = {...}
            layers: list[nn.Module] = [nn.Flatten(), ...]
            self._module = nn.Sequential(*layers)

        def train(self, train_data, config):
            # Build a DataLoader, run a few epochs of SGD,
            # then self.save(config["model_path"]).
            ...

        def test(self, test_data, config):
            # Load weights from config["model_path"], run inference.
            ...

        def save(self, path):
            torch.save({"init_kwargs": self._init_kwargs,
                        "state_dict": self._module.state_dict()}, path)

        @classmethod
        def load(cls, path):
            checkpoint = torch.load(path, weights_only=False)
            instance = cls(**checkpoint["init_kwargs"])
            instance._module.load_state_dict(checkpoint["state_dict"])
            return instance

When to reach for ``TorchModel`` instead
=========================================

The ground-up example reimplements save and load by hand. The
:class:`~kaggle_driver.integrations.torch.TorchModel` base ships exactly this
boilerplate, including a sensible default device. Subclasses only need
to implement ``build_module``, ``train``, and ``test``:

.. code-block:: python

    from kaggle_driver.integrations.torch import TorchModel

    class MNISTModel(TorchModel):
        def __init__(self, input_layer_width, output_layer_width,
                     hidden_layer_count, hidden_layer_width=None):
            self.input_layer_width = input_layer_width
            self.output_layer_width = output_layer_width
            self.hidden_layer_count = hidden_layer_count
            self.hidden_layer_width = hidden_layer_width
            super().__init__(
                input_layer_width=input_layer_width,
                output_layer_width=output_layer_width,
                hidden_layer_count=hidden_layer_count,
                hidden_layer_width=hidden_layer_width,
            )

        def build_module(self) -> nn.Module:
            # Same layer-building code as the ground-up example.
            ...

        def train(self, train_data, config):
            # Same training loop. Use self.device to place tensors.
            ...

Subclass attributes must be assigned **before** ``super().__init__`` so
``build_module`` can read them. ``build_module`` must be deterministic
from those attributes alone so :meth:`TorchModel.load` can rebuild the
module shape from the saved kwargs before populating its weights.
