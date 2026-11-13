About this guide
================

This is the documentation scaffold for the CORSIKA 8 Kokkos backend.
It is prepared for developer review and future maintenance; it is not an
official CORSIKA documentation release.

Current status
--------------

The website structure, navigation, search and chapter outlines are available.
The technical chapters have not been written or reviewed yet. An outline does
not establish that a feature is implemented or validated.

Writing a chapter
-----------------

Use the :download:`module-page template <_templates/module-page.rst.txt>`.
Record the source version checked, describe ownership and memory access, and
link to the corresponding implementation and tests. Keep implementation
descriptions separate from proposals.

Short source examples should be included from the repository with Sphinx's
``literalinclude`` directive. Public API descriptions can be included from
Doxygen XML through Breathe when the API reference is populated.

Reading and navigation
----------------------

Use the sidebar to browse the chapters, the search field to find a topic, and
the previous/next links to read the guide in sequence. The mobile menu contains
the same navigation.
