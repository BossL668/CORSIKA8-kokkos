Development and maintenance
===========================

This chapter explains how to develop features and maintain the Kokkos
integration within CORSIKA 8. It addresses three practical tasks: finding the
code to change, extending an existing component, and keeping the native and
accelerated paths consistent as C8 evolves.

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
   development/compatibility

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
   * - Follow C8 or dependency updates
     - Identify affected adapters, update dependent data and test the native
       and accelerated paths before releasing the change.
     - :doc:`development/compatibility`

Current maintenance model
-------------------------

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

Detailed cache versions and worker/file formats are collected in
:doc:`development/formats`. They are needed when those formats change, rather
than for every numerical or application edit.
