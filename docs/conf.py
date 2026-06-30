"""Sphinx configuration for the SofaBuffers Python API documentation.

Built by the ``docs.yml`` workflow with ``sphinx-apidoc`` (to generate the
per-module ``.rst`` stubs from ``src/sofab``) followed by the HTML builder, and
published to GitHub Pages.
"""

from __future__ import annotations

import os
import sys

# Make the editable ``sofab`` package importable for autodoc even when the docs
# are built without an install step.
sys.path.insert(0, os.path.abspath("../src"))

project = "SofaBuffers"
author = "SofaBuffers contributors"
copyright = "SofaBuffers contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "alabaster"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
