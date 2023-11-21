import os
import sys

sys.path.insert(0, os.path.abspath("../examples"))

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
    "sphinx_gallery.gen_gallery",
]
source_suffix = ".rst"
master_doc = "index"
project = "Kaggle Driver"
year = "2023"
author = "Christopher Priebe"
copyright = f"{year}, {author}"
version = release = "0.0.0"

autodoc_mock_imports = ["numpy", "torch"]

class ResetArgv:
    def __repr__(self):
        return 'ResetArgv'

    def __call__(self, sphinx_gallery_conf, script_vars):
        if script_vars["src_file"] == "ground_up_mnist_torch.py":
            return ["-h"]

sphinx_gallery_conf = {
     "examples_dirs": "../examples",
     "gallery_dirs": "_build/auto_examples",
     "reset_argv": ResetArgv(),
}

pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": ("https://github.com/christopherpriebe/kaggle-driver/issues/%s", "#"),
    "pr": ("https://github.com/christopherpriebe/kaggle-driver/pull/%s", "PR #"),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

if not on_rtd:  # only set the theme if we are building docs locally
    html_theme = "sphinx_rtd_theme"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
