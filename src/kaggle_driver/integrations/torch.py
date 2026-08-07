"""``TorchModel``: abstract :class:`Model` base for PyTorch users.

Install with ``pip install kaggle-driver[torch]``.

The base class deliberately ships *only* the persistence and device
boilerplate. Training and testing remain abstract because PyTorch training
loops vary too much across competitions to bake a useful default. Subclass
this and implement ``build_module``, ``train``, and ``test``.
"""

import abc
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as _error:  # pragma: no cover
    raise ImportError(
        "kaggle_driver.integrations.torch requires torch. "
        "Install with 'pip install kaggle-driver[torch]'.",
    ) from _error

from kaggle_driver.core import Model

__all__ = ["TorchModel"]


def _select_default_device() -> torch.device:
    """Return CUDA if available, else MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchModel(Model[Any, Any], abc.ABC):
    """Abstract :class:`Model` base for PyTorch.

    Handles ``state_dict`` save and load and supplies a sensible default
    device. Subclasses must implement:

    * :meth:`build_module`, which constructs the underlying ``nn.Module``.
      It must be deterministic given the constructor kwargs so that
      :meth:`load` can reconstruct the module shape before loading
      weights.
    * :meth:`train` and :meth:`test`, which drive the actual fit and
      predict loops.

    Subclasses must call ``super().__init__(**init_kwargs)`` with their
    constructor kwargs so :meth:`load` can reconstruct the instance. The
    subclass attributes used by ``build_module`` must be set before the
    ``super().__init__`` call. The typical pattern is::

        class MyModel(TorchModel):
            def __init__(self, hidden: int) -> None:
                self.hidden = hidden
                super().__init__(hidden=hidden)

            def build_module(self) -> nn.Module:
                return nn.Sequential(nn.Linear(10, self.hidden), nn.Linear(self.hidden, 1))
    """

    def __init__(self, **init_kwargs: Any) -> None:
        self._device = _select_default_device()
        self._init_kwargs: dict[str, Any] = dict(init_kwargs)
        self._module = self.build_module().to(self._device)

    @property
    def device(self) -> torch.device:
        """Return the active torch device for this model."""
        return self._device

    @property
    def module(self) -> nn.Module:
        """Return the underlying ``nn.Module``."""
        return self._module

    @abc.abstractmethod
    def build_module(self) -> nn.Module:
        """Construct and return the model's ``nn.Module``.

        Must depend only on the constructor arguments that were passed to
        ``super().__init__`` so :meth:`load` can rebuild the same shape.
        """

    def save(self, path: Path) -> None:
        """Save constructor kwargs and weights to a single file.

        The file is written via ``torch.save`` and contains a dict with
        ``"init_kwargs"`` and ``"state_dict"`` keys.

        Args:
            path: File path to write to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "init_kwargs": self._init_kwargs,
                "state_dict": self._module.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TorchModel":
        """Reconstruct a model previously written by :meth:`save`.

        Args:
            path: File path produced by an earlier :meth:`save` call.

        Returns:
            A fresh instance of ``cls`` with weights restored.
        """
        checkpoint = torch.load(path, weights_only=False)
        instance = cls(**checkpoint["init_kwargs"])
        instance._module.load_state_dict(checkpoint["state_dict"])
        return instance
