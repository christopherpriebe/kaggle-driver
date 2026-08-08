"""kaggle-driver: scaffolding for Kaggle competition workflows.

Public surface:

* :class:`Dataset`: abstract base for a competition dataset.
* :class:`Model`: abstract base for a competition model.
* :class:`KaggleInfo`: competition metadata for the Kaggle API integration.
* :func:`run`: build and dispatch a CLI for a given dataset and model set.

Optional built-in helpers live under :mod:`kaggle_driver.integrations`. They are
not re-exported here; import them directly when needed and install the
matching extra (for example, ``pip install kaggle-driver[sklearn]``).
"""

__version__ = "0.1.0"

from kaggle_driver.core import Dataset, KaggleInfo, Model
from kaggle_driver.run import run

__all__ = ["Dataset", "KaggleInfo", "Model", "__version__", "run"]
