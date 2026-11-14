"""Build the Kokkos developer guide without compiling CORSIKA or its models."""

import os
from pathlib import Path
import sys

project = "CORSIKA 8 with Kokkos"
author = "CORSIKA 8 Kokkos contributors"
copyright = "2026, CORSIKA 8 Kokkos contributors"
version = "Source: d5297c43"
release = version
language = "en"
root_doc = "index"
source_suffix = ".rst"
sys.path.insert(0, str(Path(__file__).parent / "_ext"))
extensions = ["sphinx.ext.mathjax", "sphinx.ext.graphviz", "breathe", "api_links"]
exclude_patterns = ["_build", "_includes", "_templates", "BUILDING.rst"]
templates_path = ["_templates"]
nitpicky = True

# The build helper extracts public headers and supplies its fresh XML directory.
# No model libraries or device compiler are needed for API extraction.
breathe_projects = {}
breathe_default_project = "CORSIKA8"
if os.environ.get("C8_DOCS_DOXYGEN_XML"):
    breathe_projects["CORSIKA8"] = str(
        Path(os.environ["C8_DOCS_DOXYGEN_XML"]).resolve()
    )
api_source_revision = "d5297c43c041a22b43c65c322d9b51bc7ec8e397"

html_theme = "sphinx_rtd_theme"
html_title = "CORSIKA 8 Kokkos — Documentation"
html_short_title = "Kokkos documentation"
html_static_path = ["_static"]
html_css_files = ["guide.css"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "titles_only": True,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#172e46",
}
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = True
html_search_language = "en"
html_baseurl = os.environ.get("C8_DOCS_BASEURL", "")
graphviz_output_format = "svg"

# Avoid treating network availability as a requirement for the local HTML build.
linkcheck_timeout = 15
linkcheck_retries = 1
