Developer guide
===============

This part of the documentation explains how to change the existing CORSIKA 8
with Kokkos implementation and maintain it as C8, its physics dependencies and
supported platforms evolve. It follows the development process: locate the
responsible component, implement a change, verify it on the affected execution
paths, and maintain compatibility with the surrounding framework.

The component chapters explain how the current system works. The pages here
explain how to work on that system. They distinguish existing interfaces from
recommendations for further integration with C8.

Guide structure
---------------

.. list-table::
   :header-rows: 1
   :widths: 29 45 26

   * - Chapter
     - Development responsibility
     - Main outcome
   * - :doc:`development`
     - Locate implementations and callers; modify calculations or add an
       environment, reaction or radio capability.
     - A change in the appropriate component, with its integration points
       identified.
   * - :doc:`backends`
     - Configure a development build; manage runtime and data lifetimes;
       compile and tune shared code for CPU and GPU execution.
     - An implementation that runs on the intended backend without
       duplicating the physical model.
   * - :doc:`validation`
     - Select component tests, compare physical results and measure
       performance under a defined workload.
     - Evidence supporting the specific correctness and performance claims
       of the change.
   * - :doc:`decisions`
     - Follow upstream and dependency changes; maintain interfaces, data
       formats and native/accelerated behavior together.
     - Compatibility with C8 as the implementation evolves.

The separate :doc:`api` provides declarations and member documentation. Use it
to look up a signature or type after locating the responsible component.

Development workflow
--------------------

1. **Identify the behavior to change.** Trace the application input to the
   responsible preparation, numerical or execution function. The file map
   and continuous-energy-loss example in :doc:`development/workflow` show
   how to locate its callers and tests.
2. **Prepare a reproducible build.** Keep native and accelerated build
   directories separate. Record the C8 revision, dependency versions and
   backend configuration. :doc:`backends/build` explains the targets and
   toolchain settings; :doc:`usage/installation` provides installation
   commands.
3. **Implement at the appropriate layer.** Change data conversion when an
   input changes, the numerical implementation when a calculation changes,
   or the scheduler when work assignment changes. Reuse existing interfaces
   and shared functions; :doc:`development/extensions` and
   :doc:`development/shared-algorithms` give concrete examples.
4. **Validate the affected behavior.** Start with controlled-input tests,
   then exercise the connected execution path. Use complete showers when
   the change affects physical distributions or event-level behavior.
   :doc:`validation` explains how to select the comparison and timing scope.
5. **Prepare the change for review and maintenance.** Explain the resulting
   behavior, affected callers and validation evidence. Update the relevant
   documentation. If an interface, dependency or persisted record changes,
   follow :doc:`development/compatibility` as well.

Common development tasks
------------------------

.. list-table::
   :header-rows: 1
   :widths: 41 59

   * - Task
     - Starting point
   * - Fix an existing transport calculation
     - :doc:`development/workflow`, followed by
       :doc:`validation/numerics` for the reference and tolerance.
   * - Add a material profile, shape or reaction
     - :doc:`development/extensions` for preparation, evaluation and
       integration changes.
   * - Reduce duplicate native and Kokkos code
     - :doc:`development/shared-algorithms` for the shared boundary-selection
       example and extraction of another common calculation.
   * - Support or optimize an execution backend
     - :doc:`backends/device-code`, with :doc:`backends/runtime` for resource
       ownership and :doc:`validation/performance` for measurements.
   * - Update C8, PROPOSAL or Kokkos
     - :doc:`development/compatibility` for the affected adapters, prepared
       data and regression checks.
   * - Simplify integration with the official framework
     - :doc:`decisions/interfaces` for application and data boundaries;
       :doc:`decisions/execution-models` for scheduling and reproducibility.

Maintenance principles
----------------------

Native C8 constructs the scene and configured physics models. Preparation
adapters make the required data available to numerical functions. Execution
code manages storage, queues and scheduling. Keeping these responsibilities
separate makes it possible to change a calculation without redesigning the
application or implementing it separately for CUDA and HIP.

OpenMP, CUDA and HIP already use common Kokkos numerical implementations.
Sharing with the native C8 path is more selective: some functions share
source, while others use exported native data with separate calculations.
Maintenance must therefore cover both shared functions and adapters between
separately maintained implementations. The integration chapter identifies
the remaining choices without treating them as decisions already adopted by
the CORSIKA project.
