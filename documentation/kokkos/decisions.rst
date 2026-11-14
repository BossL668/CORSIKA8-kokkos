Design decisions and open questions
===================================

The integration combines native C8 scene and model construction with numerical
data and execution services suitable for Kokkos. The important design
boundaries are where objects are borrowed, data are converted, calculations
are dispatched and results return to C8. Those boundaries determine both
portability and the amount of code that must be maintained together.

This chapter separates the design visible in the implementation from options
for future integration. It records consequences and review questions rather
than attributing undocumented intentions to earlier changes.

.. note::

   Implementation baseline: `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   **Current implementation** means the behavior found at this revision.
   **Integration option** means a candidate for discussion, not an implemented
   or approved upstream interface. Performance and validation requirements
   below are proposed evaluation criteria, not new benchmark results.

.. toctree::
   :maxdepth: 1

   decisions/interfaces
   decisions/execution-models

Decision map
------------

The component chapters document the actual calls, parameters and ownership.
The pages in this chapter explain the design choices those interfaces expose.
The identifiers below organize this guide; they are not upstream issue numbers.

.. list-table::
   :header-rows: 1
   :widths: 14 39 47

   * - Topic
     - Current implementation
     - Question for further integration
   * - D1: application integration
     - ``CascadeExtension`` configures resources and binds native objects;
       ``CascadeModule`` and ``CascadeEvent`` provide run/event lifetimes.
     - Which services belong in a common C8 execution interface, and which
       remain Kokkos-specific? See :ref:`decision-application-interface`.
   * - D2: numerical data
     - Native particle and environment objects are converted into explicitly
       owned numerical storage and borrowed views.
     - Should native and accelerated algorithms share a common data-access
       contract, and how is its supported domain maintained?
       See :ref:`decision-data-representation`.
   * - D3: physics implementation
     - Native tables and CPU model calls coexist with device-callable
       numerical routines. Some framework algorithms already share a body.
     - Which calculations can share source across paths, and which need
       an external-library adapter? See :ref:`decision-shared-calculations`.
   * - D4: execution scheduling
     - Resident particle batches use shared stage interfaces. Host routing
       and process coordination manage completion and event resources.
     - Where should execution policy vary without duplicating physics?
       See :ref:`decision-execution-policy`.
   * - D5: reproducibility
     - Device random keys include particle history; persistent CPU workers
       continue their streams. Scheduling and process lifetime matter.
     - Which changes require exact replay, numerical agreement or statistical
       agreement? See :ref:`decision-reproducibility-policy`.
   * - D6: public contracts
     - Headers, exported targets, caches, worker formats and outputs have
       distinct compatibility rules.
     - Which contracts should receive an explicit stability commitment, and
       which remain internal? See :ref:`decision-interface-stability`.

The integration does not currently reduce to one backend switch. In addition
to cascade creation, ``CascadeExtension`` participates in CLI registration,
model/writer construction, output registration and resource preparation.
Making the application smaller should therefore start by assigning those
responsibilities, rather than hiding the same parameter list behind a new
name. The actual class declaration is in `CascadeExtensionOptions.hpp`_; its
application use is documented in :doc:`application`.

.. _decision-interface-stability:

Interface stability and maintenance ownership
---------------------------------------------

The current ``corsika/kokkos`` headers and their included implementations are
installed for consumers. This does not give every exported symbol or record
an independent stability guarantee. Similarly, a cache version, CMake package
version and worker protocol version cover different contracts. The existing
checks are documented in :doc:`development/compatibility`.

For upstream integration, agree on the intended consumers of each interface:

* **Applications:** scene/model construction, execution selection and event
  lifecycle. Changes need native-only and accelerated application checks.
* **Model implementations:** numerical inputs, results, domains and units.
  Changes need model references and physical validation as appropriate.
* **Execution implementations:** memory ownership, queue capacities,
  completion and error propagation. Changes need backend and lifetime tests.
* **Analysis and operational tools:** output schemas, timing definitions,
  metadata and completion records. Changes need reader/merger compatibility.

A narrow public surface can reduce compatibility obligations, but it must
still expose the capabilities required by its consumers. For example, a host
execution interface may accept borrowed spans only if copying or completion
rules make their lifetime unambiguous. A device view can remain a small value
type while its owner handles allocation and synchronization. These are
different responsibilities even when their names refer to the same data.

The recommended maintenance agreement is to name the responsible component
and reviewer role for each contract. No individuals or review groups are
assigned by this guide. Discuss deprecation and migration before replacing a
public entry point; record whether old applications, caches and output
readers remain supported. A successful local application build cannot settle
all of those compatibility questions.

Review sequence for an integration proposal
-------------------------------------------

The following sequence keeps interface discussion connected to executable
evidence. It is a proposed workflow, not an additional repository approval
policy.

1. **Describe the boundary.** Identify the existing caller, producer, consumer,
   ownership and result contract. Link a pinned declaration and implementation.
   State the concrete limitation the change addresses.
2. **Compare implementable alternatives.** Include retaining the current path
   when it remains viable. Describe affected files, dependencies, application
   changes and failure behavior. Distinguish a scheduling change from a
   numerical or physical change.
3. **Prototype one complete use path.** Connect configuration, preparation,
   execution and outputs. A short application snippet is insufficient if
   native fallback, final batches or event cleanup remain unconnected.
4. **Evaluate the claimed scope.** Check affected interfaces and installed
   consumers, numerical/reference results and complete events where needed.
   Compare performance on matched workloads and identify cold/warm lifecycle
   costs. Use :doc:`validation` rather than a single generic tolerance.
5. **Record the reviewed decision.** Link the implementation, evidence and
   review outcome. State limitations, remaining work and migration. Mark a
   superseded record explicitly rather than rewriting its historical result.

For example, sharing one boundary-selection body requires checking both the
native and numerical access adapters. It does not imply that all shape
intersection solvers have become identical. The current
`NextEnvironmentBoundaryBody.inl`_ illustrates this distinction;
:doc:`environment/queries` specifies the shared and separate parts.

Decision record template
------------------------

Use the :download:`design-decision template
<_templates/design-decision.rst.txt>` to capture a concrete proposal. Its
fields distinguish implementation status, reasoning, alternatives and
evidence. An accepted status should cite the relevant review; a source
inspection alone establishes observed behavior, not consensus.

Keep the full numerical and performance evidence with the test report, then
link the relevant result from the decision. Record the source revision,
dependencies, device and execution mode. This allows later maintainers to
revisit a decision when the constraints change without treating an old
timing result as a permanent property of the algorithm or architecture.

.. _CascadeExtensionOptions.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeExtensionOptions.hpp
.. _NextEnvironmentBoundaryBody.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl
