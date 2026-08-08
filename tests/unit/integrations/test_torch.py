"""Unit tests for :class:`kaggle_driver.integrations.torch.TorchModel`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from kaggle_driver.integrations.torch import (  # noqa: E402
    TorchModel,
    _select_default_device,
)


class _TinyPerceptron(TorchModel):
    """Two-layer perceptron for save/load roundtrip testing.

    Subclass attributes are set *before* ``super().__init__`` so they are
    available when the base class calls ``build_module``.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(in_features=in_features, out_features=out_features)

    def build_module(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.in_features, 4),
            nn.ReLU(),
            nn.Linear(4, self.out_features),
        )

    def train(
        self,
        train_data: Mapping[str, tuple[Any, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del train_data, config
        return {}

    def test(
        self,
        test_data: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        del test_data, config
        return {}, {}


def test_save_load_preserves_weights(tmp_path: Path) -> None:
    """Test ``save`` then ``load`` reconstructs the module and restores its weights."""
    model = _TinyPerceptron(in_features=3, out_features=2)
    sample = torch.randn(1, 3).to(model.device)
    expected = model.module(sample).detach()
    artifact = tmp_path / "tiny_perceptron.pt"

    model.save(artifact)
    restored = _TinyPerceptron.load(artifact)

    assert restored.in_features == 3
    assert restored.out_features == 2
    actual = restored.module(sample.to(restored.device)).detach()
    assert torch.allclose(expected.cpu(), actual.cpu())


def test_device_reports_a_torch_device() -> None:
    """Test ``device`` returns a torch device."""
    model = _TinyPerceptron(in_features=2, out_features=1)

    assert isinstance(model.device, torch.device)


def test_select_default_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CUDA wins device selection when torch reports it available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _select_default_device() == torch.device("cuda")


def test_select_default_device_uses_mps_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test MPS is selected when CUDA is unavailable but MPS is available."""
    if getattr(torch.backends, "mps", None) is None:
        pytest.skip("torch build has no MPS backend")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert _select_default_device() == torch.device("mps")


def test_select_default_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CPU is the fallback when neither CUDA nor MPS is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert _select_default_device() == torch.device("cpu")
