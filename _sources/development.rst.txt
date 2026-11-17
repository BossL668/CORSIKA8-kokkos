.. _development-and-maintenance:

Code development and extension
==============================

This chapter covers changes to the implementation: modifying an existing
function, adding a capability and sharing calculations with native C8.
It provides file locations, examples and the connected tests for each task.
For the complete workflow, start with :doc:`developer-guide`; maintenance
across upstream and dependency updates is covered in :doc:`decisions`.

The maintenance objective is to keep physical models and numerical rules
consistent while allowing different data layouts and execution strategies.
A change to a formula should not require separate CUDA and HIP implementations.
A change to the application should not require moving particle scheduling or
device-memory management into ``main``.

.. note::

   Source examples use the guide's `d5297c43 baseline
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   The pages distinguish existing implementation from recommended development
   steps. The organization of an upstream integration remains subject to
   review with the CORSIKA maintainers.

Development guide
-----------------

.. toctree::
   :maxdepth: 1
   :hidden:

   development/workflow
   development/extensions
   development/shared-algorithms

.. list-table::
   :header-rows: 1
   :widths: 31 42 27

   * - Development task
     - What this chapter explains
     - Start here
   * - Modify or fix existing code
     - Locate the public interface, implementation, callers and relevant tests.
     - :doc:`development/workflow`
   * - Add an environment or a physics capability
     - Extend preparation, numerical evaluation and integration in the existing
       modules, with concrete examples and file locations.
     - :doc:`development/extensions`
   * - Reduce duplicated algorithms
     - Identify what is already shared and extract calculations that native C8
       and Kokkos can call through different data-access interfaces.
     - :doc:`development/shared-algorithms`

Implementation responsibilities
-------------------------------

The application builds the C8 environment, physics processes and observers.
Host adapters convert the inputs needed by Kokkos. Numerical functions perform
the supported calculations; execution and scheduling code decide where and
when those functions run. The component chapters describe these layers in
:doc:`application`, :doc:`environment`, :doc:`physics-data`, :doc:`transport`
and :doc:`execution`.

Two kinds of code sharing must be distinguished:

* **Between Kokkos backends:** OpenMP, CUDA and HIP instantiate common numerical
  implementations. Device-specific launch settings and runtime operations can
  differ without creating another physical model.
* **Between native C8 and Kokkos:** some algorithms already share source, such
  as the environment boundary-selection body. Other calculations, including
  access to exported PROPOSAL tables, still have separate implementations.
  Sharing table coefficients does not by itself remove that maintenance work.

The recommended direction is to share numerical calculations where practical,
keep data conversion explicit, and keep scheduling outside those calculations.
:doc:`development/shared-algorithms` describes how to do this incrementally,
without replacing the native application or porting an entire external
physics library at once.

A typical contribution
----------------------

1. Identify the changed behavior and locate its implementation using
   :doc:`development/workflow`.
2. Update the responsible component and any affected input conversion or
   caller. Keep unrelated physics and scheduling changes separate.
3. Run the relevant component comparison, then the integration or shower
   comparison needed for that change. :doc:`validation` gives the test map
   and comparison methods.
4. Describe the changed behavior, affected execution paths and test results
   in the review. Update the component documentation and examples with the code.

Changes to a public interface, external dependency or persisted data also
need the compatibility review described in :doc:`development/compatibility`.
Detailed versioned records are collected in :doc:`development/formats`.
