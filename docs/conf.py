"""Sphinx configuration for the pyBBTD documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pyBBTD"
author = "Saulo Barreto"
copyright = "2025, Saulo Barreto"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# Notebooks are shown with the outputs saved in them (see scripts/run_notebooks.py).
nb_execution_mode = "off"
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "sphinx_book_theme"
html_title = "pyBBTD"
html_theme_options = {
    "repository_url": "https://github.com/scbarreto/pyBBTD",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
}
