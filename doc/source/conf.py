# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from codecs import open
import os
import sys
from unittest import mock
from pkg_resources import parse_requirements
sys.path.insert(0, os.path.abspath('../../'))


# Read in the IMAGINE requirements
with open('../../requirements.txt', 'r') as file:
    requirements = file.read().splitlines()
with open('../requirements_RTD.txt', 'r') as file:
    docs_requirements = file.read().splitlines()

# Parse the requirements
parsed_reqs = parse_requirements(requirements)
parsed_docs_reqs = parse_requirements(docs_requirements)

# Determine the names of all requirements
req_names = set(map(lambda x: x.name, parsed_reqs))
docs_req_names = set(map(lambda x: x.name, parsed_docs_reqs))

# Take the difference between req_names and docs_req_names
names = req_names.difference(docs_req_names)

# Create mock modules for all modules IMAGINE needs but the docs do not
for mod_name in names:
    sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = 'IMAGINE'
copyright = '2019-2020, IMAGINE Consortium'
author = 'IMAGINE Consortium'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'nbsphinx_link',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

#html_logo = 'logo.png'


dd_function_parentheses = True
add_module_names = True
numfig = True

# Reference formatting
numfig_format = {'figure': "Fig. %s"}


# ------------------------------------------------------------------
# Autodoc configuration
autodoc_default_options = {'members': None,
                           'special-members': '__call__'}

autodoc_member_order = 'groupwise'
autodoc_inherit_docstrings = True
autosectionlabel_prefix_document = True

# Napoleon configuration
napoleon_include_private_with_doc = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True


intersphinx_mapping = {'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                      'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                      'astropy': ('https://docs.astropy.org/en/stable/', None),
                      'python': ('https://docs.python.org/3', None),
                      'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                      'ultranest': ('https://johannesbuchner.github.io/UltraNest/', None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#


html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
