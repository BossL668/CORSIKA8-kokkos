Building the Kokkos developer guide
===================================

Run from the repository root. Python 3.11 or newer is required by the pinned
Sphinx version. This builds documentation only; no CORSIKA binary, FLUKA,
PROPOSAL or GPU toolchain is required.

The chapters use ``literalinclude`` to read excerpts from tracked C++ and CMake
files. In a sparse checkout, retain the referenced files under
``applications``, ``corsika``, ``src``, ``tests`` and ``cmake``, together with
the root build files, ``documentation`` and the documentation build tool.
No compilation is involved. Stored SVG diagrams are ready to use; their
adjacent DOT sources are only needed when editing the diagrams, with Graphviz's
``dot -Tsvg`` command.

Local preview::

   python3 -m venv build/kokkos-docs-env
   build/kokkos-docs-env/bin/python -m pip install -r documentation/kokkos/requirements.txt
   build/kokkos-docs-env/bin/python tools/build_kokkos_docs.py
   build/kokkos-docs-env/bin/python -m http.server 8765 --bind 127.0.0.1 --directory build/kokkos-docs/html

Open http://localhost:8765/ . Stop the preview with Ctrl-C. No data is uploaded.
The generated site can also be opened via build/kokkos-docs/html/index.html;
HTTP preview is recommended for testing search.

The builder treats documentation warnings as errors and checks internal HTML
links and anchors. To use another output location::

   build/kokkos-docs-env/bin/python tools/build_kokkos_docs.py --output /tmp/c8-kokkos-guide

Public hosting
--------------

The initial site is published from the ``gh-pages`` branch. That branch contains
only the checked static HTML output, including ``.nojekyll``; editable sources
stay in ``documentation/kokkos`` on the main development branch. In repository
Settings > Pages, select "Deploy from a branch", ``gh-pages``, and the root
folder. After editing, rebuild and validate the guide before updating that
publication branch. A source commit alone does not update the website.

An optional Actions workflow is provided in
``_templates/github-pages-workflow.yml.txt``. To enable automatic build checks,
copy it to ``.github/workflows/kokkos-docs.yml`` using a GitHub credential with
workflow permission. To also deploy through Actions, switch the Pages source
to GitHub Actions and run that workflow on the default branch with ``publish``
enabled. The workflow template itself is not active.

For Read the Docs, use documentation/kokkos/conf.py as the Sphinx configuration
and documentation/kokkos/requirements.txt as the dependency list. This provides
the same source content without a second copy of the guide.

The chapters are ordinary reStructuredText and are linked from the parent
documentation/index.rst. When incorporating this guide in the official site,
reuse its theme and review the standalone presentation settings separately.

API reference
-------------

Set C8_DOCS_DOXYGEN_XML to an existing Doxygen XML directory when adding Breathe
API directives. The initial scaffold does not generate API pages. This keeps
the preview independent of a full project build.
