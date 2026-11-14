About this guide
================

This guide is intended for users, developers and maintainers of CORSIKA 8 with Kokkos.
Its purpose is to explain the implementation and provide a reference for
continued development, review and maintenance as the code, dependencies and
supported platforms evolve.

Current status
--------------

The overview and user guide describe the current application, installation,
configuration and output. Their source baseline is recorded in the
:doc:`usage/index`. The :doc:`application` chapter now follows concrete function
calls, data contracts and event execution, using excerpts included from source.
The :doc:`execution` chapter covers the hybrid loop, batch contracts and
process coordination, with file maps and data-flow diagrams. The
:doc:`particles` chapter documents physical-value conventions, complete-state
transfer and resident storage, with source excerpts and a test map. The
:doc:`environment` chapter covers scene export, ownership, numerical queries,
shared boundary selection and model-extension requirements. The
:doc:`physics-data` chapter covers native model preparation, table
representation, resident ownership and query/extension contracts. The
:doc:`transport` chapter follows candidate sampling, step-limit competition,
continuous processes and final-state/CPU completion contracts. The remaining
architecture and maintenance chapters are still outlines; an
outline does not establish that a feature is implemented or validated.
This site is not an official CORSIKA documentation release.

Writing a chapter
-----------------

Use the :download:`module-page template <_templates/module-page.rst.txt>`.
Record the source version checked, describe ownership and memory access, and
link to the corresponding implementation and tests. Keep implementation
descriptions separate from proposals.

Use concise technical headings that identify the component, lifecycle stage or
interface contract. Define responsibilities, inputs, outputs and ownership in
direct terms, retaining the original source identifiers. Prefer established
software terminology over conversational questions or reading instructions.

Short source examples should be included from the repository with Sphinx's
``literalinclude`` directive. Public API descriptions can be included from
Doxygen XML through Breathe when the API reference is populated.

Reading and navigation
----------------------

Use the sidebar to browse the chapters, the search field to find a topic, and
the previous/next links to read the guide in sequence. The mobile menu contains
the same navigation.
