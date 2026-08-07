"""End-to-end TorchModel save and load roundtrip on a tiny synthetic problem."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn, optim  # noqa: E402

from kaggle_driver.integrations.torch import TorchModel  # noqa: E402


class _LinearRegressor(TorchModel):
    """One-layer linear regressor that fits its data in five epochs."""

    def __init__(self, in_features: int) -> None:
        self.in_features = in_features
        super().__init__(in_features=in_features)

    def build_module(self) -> nn.Module:
        return nn.Linear(self.in_features, 1)

    def train(
        self,
        train_data: Mapping[str, tuple[Any, Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        epochs = int(config.get("epochs", 50))
        learning_rate = float(config.get("lr", 0.05))
        inputs = torch.stack(
            [torch.as_tensor(x, dtype=torch.float32) for x, _ in train_data.values()],
        ).to(self.device)
        targets = torch.stack(
            [torch.as_tensor([y], dtype=torch.float32) for _, y in train_data.values()],
        ).to(self.device)
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(self.module.parameters(), lr=learning_rate)
        losses: list[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.module(inputs)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        return {"final_loss": losses[-1]}

    def test(
        self,
        test_data: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, float], Mapping[str, Any]]:
        del config
        predictions: dict[str, float] = {}
        for example_id, features in test_data.items():
            tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            predictions[example_id] = float(self.module(tensor).item())
        return predictions, {}


@pytest.mark.integration
@pytest.mark.slow
def test_torch_model_train_save_load_predict(tmp_path: Path) -> None:
    """Test training, persisting, reloading, and predicting yields matching outputs."""
    train_data = {str(i): ([float(i) / 10, float(i * 2) / 10], float(i * 3) / 10) for i in range(8)}
    test_data = {"a": [0.1, 0.2], "b": [0.3, 0.6]}
    model = _LinearRegressor(in_features=2)

    statistics = model.train(train_data, config={"epochs": 200, "lr": 0.05})

    assert statistics["final_loss"] < 1.0

    artifact = tmp_path / "linear_regressor.pt"
    model.save(artifact)
    restored = _LinearRegressor.load(artifact)

    original_predictions, _ = model.test(test_data, config={})
    restored_predictions, _ = restored.test(test_data, config={})

    for key in original_predictions:
        assert abs(original_predictions[key] - restored_predictions[key]) < 1e-5
