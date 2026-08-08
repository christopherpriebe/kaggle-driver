==================
Core API reference
==================

.. automodule:: kaggle_driver
    :members: Dataset, Model, KaggleInfo, run
    :undoc-members:
    :show-inheritance:

Immutability helper
===================

Mappings that cross a framework boundary are frozen so no later stage can
mutate data an earlier one still holds. Implementations of ``Dataset`` and
``Model`` may return an ordinary ``dict``; the framework freezes it before
passing it on, and everything handed back to the caller is an
``immutabledict``.

.. autofunction:: kaggle_driver.core.freeze_mapping
