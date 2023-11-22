# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

import pytest
from utils import DummyTarget
from kaggle_driver.dataset.target import Target


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        "fish",
        [0, 1, 2],
        {"a": 0, "b": 1},
        (2, 3, 5),
        DummyTarget("fish")
    ],
)
def test_target_holds_value(value) -> None:
    """Tests that the target holds the correct value.
    """
    dummy_target = DummyTarget(value)
    assert isinstance(dummy_target, Target)
    assert id(dummy_target.value) == id(value)
    assert dummy_target.value == value
