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

# ``sofab/__init__.py`` re-exports Visitor, so autodoc documents it twice — once as
# ``sofab.Visitor`` (package page) and once as ``sofab.visitor.Visitor`` (module
# page). With ``autodoc_typehints = "description"`` the annotation of
# ``Decoder.drive(visitor: Visitor)`` renders as the bare name ``Visitor``, which
# then has two equally good targets and warns ("more than one target found for
# cross-reference 'Visitor'"). Rendering that annotation as the fully-qualified
# public name resolves it to exactly one object, so the build is warning-free and
# the parameter type links to the page users are meant to read.
autodoc_type_aliases = {"Visitor": "sofab.Visitor"}

html_theme = "alabaster"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
