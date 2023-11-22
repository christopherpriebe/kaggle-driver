# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

import pytest
from utils import DummyInput
from kaggle_driver.dataset.input import Input


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        "fish",
        [0, 1, 2],
        {"a": 0, "b": 1},
        (2, 3, 5),
        DummyInput("fish")
    ],
)
def test_input_holds_value(value) -> None:
    """Tests that the input holds the correct value.
    """
    dummy_input = DummyInput(value)
    assert isinstance(dummy_input, Input)
    assert id(dummy_input.value) == id(value)
    assert dummy_input.value == value
