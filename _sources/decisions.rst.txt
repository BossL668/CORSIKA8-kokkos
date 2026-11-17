.. _design-decisions-and-open-questions:

C8 integration and long-term maintenance
========================================

Maintaining this implementation means keeping native C8 behavior, preparation
adapters, numerical calculations and execution code consistent as each
changes. This chapter covers routine updates first, then the interface
changes that could reduce duplicated work during further integration with
the official framework.

The implementation descriptions refer to the guide's ``d5297c43`` baseline.
Recommendations for future interfaces are identified as such; they are not
an interface agreement already adopted by the CORSIKA project.

Maintenance tasks
------------------

.. list-table::
   :header-rows: 1
   :widths: 29 44 27

   * - Change
     - Maintenance work
     - Detailed guide
   * - Update C8 or an external dependency
     - Inspect changed native behavior, adjust the affected adapters and
       verify native and accelerated execution together.
     - :doc:`development/compatibility`
   * - Change a prepared-data or worker format
     - Review readers, writers, cache identities and reused-session behavior;
       document migration or regeneration requirements.
     - :doc:`development/formats`
   * - Refine the application interface
     - Separate configuration and preparation from execution construction;
       keep native objects and process-sequence semantics intact.
     - :doc:`decisions/interfaces`
   * - Reduce duplicate calculations
     - Extract a common numerical function where the input domain and
       conventions agree; retain adapters for storage and native model calls.
     - :doc:`development/shared-algorithms`
   * - Change scheduling or worker reuse
     - Preserve complete particle state, supported reactions, output
       accounting and a documented random-stream policy.
     - :doc:`decisions/execution-models`

.. toctree::
   :maxdepth: 1
   :hidden:

   development/compatibility
   decisions/interfaces
   decisions/execution-models

Incremental integration with C8
-------------------------------

An incremental integration can use the existing component boundaries rather
than replace the native framework. The following is a recommended order for
reviewing changes, not a scheduled upstream roadmap.

1. **Define the application boundary.** Agree which native scene, models,
   stack and outputs are supplied to execution construction. Keep preparation
   services separate from particle scheduling. Preserve a usable native path
   and its defaults while reviewing the accelerated alternative.
2. **Define data access and ownership.** Keep one C8 scene construction and
   explicit conversion into supported numerical data. Specify units,
   coordinates, complete particle state and the lifetime of borrowed objects.
   Shared value types or accessors can be introduced independently of the
   storage layout.
3. **Share calculations where practical.** Use the common boundary-selection
   body as an existing example. Extract another calculation only after its
   inputs, result, domain and reference are defined. External model adapters
   remain necessary where the dependency does not provide a portable core.
4. **Keep execution replaceable.** Maintain the distinction between a physical
   operation and the queues, policies or workers that schedule it. Evaluate
   scheduling changes through both correctness tests and representative
   event measurements.

:doc:`decisions/interfaces` explains the current application and data boundary
and the alternatives. :doc:`decisions/execution-models` explains which
execution responsibilities must survive such a refactor. Both link to the
current implementation; proposed interfaces are not presented as existing
functions.

.. _decision-interface-stability:

Interface stability and maintenance ownership
---------------------------------------------

For each public interface, identify its callers and the component responsible
for maintaining it. Application bindings affect scene/model setup and event
lifecycle. Numerical interfaces affect domains, units, results and failure
status. Execution interfaces affect storage, completion and synchronization.
Persisted formats also affect readers outside the simulation process.

Review a change through those callers. Installed headers, CMake targets,
prepared-data caches, worker records and analysis outputs do not share one
compatibility rule. A cache may require regeneration while an application API
can remain unchanged. Conversely, compiling an application does not verify
that an existing output reader still interprets a changed field correctly.
See :doc:`development/compatibility` and :doc:`development/formats` for the
concrete checks.

The current ``corsika/kokkos`` headers are installed for consumers, but their
installation alone is not a stability commitment for every helper or record.
The intended public surface, deprecation policy and reviewer responsibilities
need agreement during upstream integration. This guide does not assign those
responsibilities to individual maintainers.

Reviewing and releasing a maintained version
--------------------------------------------

A review should explain the changed behavior, affected native and accelerated
paths, and the evidence supporting the change. Keep the source revision,
build configuration and comparison inputs with the results. Document an
intentional numerical or random-stream change separately from an interface
refactor so reviewers can assess each claim.

Before publishing a maintained version, check the native configuration, the
affected CPU/GPU configurations and an installed consumer. Include event
reuse and coordination when those paths change. Update the relevant API,
component explanation and runnable example together. The detailed update
procedure is in :doc:`development/compatibility`; the test and measurement
methods are in :doc:`validation`.
