import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dnamite'
copyright = '2025, Mike Van Ness'
author = 'Mike Van Ness'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # This enables autodoc
    'sphinx_rtd_theme',
    'nbsphinx',  # Enable nbsphinx for .ipynb support
    "sphinx.ext.napoleon", # NumPy style docstrings
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

add_function_parentheses = False

html_logo = '../dynamite.png'

# autodoc_default_options = {
#     'members': True,
#     'show-inheritance': True,
#     'inherited-members': False,
# }
