# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys, os
sys.path.insert(0, os.path.abspath('../../'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mypackage'
copyright = '2022, Adwaye Rambojun'
author = 'Adwaye Rambojun'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

autodoc_typehints = 'signature'

autoclass_content = 'class'


templates_path = ['_templates']
exclude_patterns = []

language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# import sphinx_readable_theme
html_theme = 'renku'
# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme_options = {
#     "stickysidebar": "true",
#     "rightsidebar": "false"
# }
# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"
# }
html_static_path = ['_static']
autodoc_mock_imports = ["torch","torchvision","Augmentor","joblib","pandas"]
