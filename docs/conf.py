"""Sphinx configuration for the kaggle-driver documentation."""

from __future__ import annotations

import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

sys.path.insert(0, str(Path("../examples").resolve()))

project = "Kaggle Driver"
author = "Christopher Priebe"
copyright = f"2023-2026, {author}"
release = _pkg_version("kaggle-driver")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

source_suffix = ".rst"
master_doc = "index"
templates_path = ["."]
pygments_style = "trac"

html_theme = "sphinx_rtd_theme"
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = False

extlinks = {
    "issue": (
        "https://github.com/christopherpriebe/kaggle-driver/issues/%s",
        "#%s",
    ),
    "pr": (
        "https://github.com/christopherpriebe/kaggle-driver/pull/%s",
        "PR #%s",
    ),
}
