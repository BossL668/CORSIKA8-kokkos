Application and data interface boundaries
=========================================

An integration interface must connect the native simulation to an execution
implementation without making application code responsible for particle
scheduling or device storage. The decision is broader than the number of
calls in ``main``: it determines ownership, optional dependencies and the
contracts that new applications must implement.

This page distinguishes the implementation at `d5297c43`_ from alternatives
for discussion. The alternatives are not approved upstream interfaces and
are not additional working modes in the current application.

.. _decision-application-interface:

Application insertion point
---------------------------

**Current implementation.** ``main`` constructs the native scene, models,
writers and ``make_sequence(...)``. ``select_execution(...)`` selects direct
execution or a coordinator before this setup. On the direct path,
``CascadeExtension::geometry(...)`` and ``execution(...)`` create borrowed
bindings and a persistent ``CascadeModule``. Each event obtains a
``CascadeAdapter`` through ``event.cascade(...)`` and starts it with
``EAS.run()``. Its native branch constructs the original C8 ``Cascade``;
its accelerated branch enters the hybrid driver. See `CascadeExtension.inl`_,
`CascadeDriver.inl`_ and :doc:`../application/event-execution`.

The extension currently also registers options, configures runtime resources,
constructs certain writers and interaction adapters, and manages diagnostics.
It is therefore an application integration component, not only an execution
policy. Keeping ``EAS.run()`` does not mean that the type of ``EAS`` or every
construction step is unchanged. The responsibilities are mapped in
:doc:`../application/setup`.

The following choices address where this integration should live:

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Choice
     - Benefit
     - Cost or unresolved contract
   * - Retain the current extension and adapter
     - Keeps scheduling, worker handling and backend initialization in the
       library while preserving the native scene and process sequence.
     - The extension exposes several responsibilities. The air-shower
       application includes this layer even when selecting native
       execution; its optional-dependency boundary needs regression coverage.
   * - Separate preparation services from execution construction
     - Keep option parsing, writer creation and diagnostics in explicit
       services; concentrate execution selection near sequence and cascade
       construction. Reuse the existing driver below that boundary.
     - A small factory still needs complete scene, model and output inputs.
       A typed context can organize them, but does not eliminate ownership,
       lifetime or capability requirements. This would be an interface
       refactor, not removal of those requirements.
   * - Add an execution policy to native cascade construction
     - Native and accelerated execution could be selected through a common
       framework-owned construction contract.
     - Requires an upstream extension point for batching, CPU completion,
       output lifecycle and synchronization. The existing scalar cascade
       cannot acquire those semantics from a policy name alone.

A process-sequence replacement alone is insufficient for the current design:
the hybrid driver routes stack entries, submits resident batches and imports
returned states. A new sequence-level interface would have to define how
these operations interact with the surrounding cascade loop. Conversely,
the current adapter does not prove that upstream must adopt this exact
wrapper. See :doc:`../execution/hybrid-loop` for the behavior any replacement
must preserve.

**Review evidence.** Compare native-only and accelerated builds, application
defaults, the process sequence, output hooks and event completion. Exercise
direct execution, worker execution and multiple events. A shorter ``main``
is a readability result; it is not evidence of equivalent physical behavior.

.. _decision-data-representation:

Native objects and numerical data
---------------------------------

**Current implementation.** Host application bindings accept native C8
objects and typed quantities. ``CascadeGeometry`` and related context bundles
borrow the scene, models, tracking and output objects; their owners must
outlive their use. Prepared numerical data has separate owners. For example,
``EnvironmentSnapshot`` owns exported records while its topology retains
host mappings to native nodes; ``EnvironmentStorage<TExecutionSpace>`` owns
uploaded arrays. See `CascadeContext.hpp`_ and `EnvironmentSnapshot.hpp`_.

Particle transfer converts physical values into the backend's documented
units and common coordinate system. Complete transfer also preserves history,
ancestry, step identity and logical-node state. Restoring a particle is not
equivalent to copying only its energy, position and direction. The production
conversion boundary is `RouterParticleConversion.inl`_; its precise contract
is in :doc:`../particles/conversion`.

There are three distinct questions when considering greater type sharing:

* **Value representation:** can a quantity or vector type carry the required
  numerical components and unit semantics?
* **Memory access:** are its fields and every referenced allocation accessible
  and alive in the selected execution space?
* **Operations:** can the entire called function chain execute on the device,
  including coordinate conversion and any dynamic dispatch?

Units, classes and pointers are not intrinsically incompatible with GPU
execution. The current native objects also depend on host allocations,
coordinate-system handles and host implementations. Sharing their names or
copying their bytes does not satisfy the latter two requirements. Managed
memory would address some accessibility questions but would not make host
methods or host virtual-function tables device-callable. The device-access
constraints are documented in :doc:`../environment/extension` and
:doc:`../backends/device-code`.

Two maintenance directions can be evaluated independently:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Direction
     - Potential benefit
     - Required work
   * - Retain explicit numerical conversion
     - Keeps device ownership and unit conventions visible; native model
       interfaces remain usable without device compilation.
     - Maintain conversion contracts and tests for every added field. Logical
       state and physical values must remain complete across CPU completion.
   * - Share small value types and numerical accessors
     - Could reduce duplicated unit conventions and calculations while keeping
       separate storage owners and native scene objects.
     - Audit dependencies and callable operations, define conversions and
       layout, and test each backend. Reusing a value type does not imply
       reusing the host object graph or native stack implementation.

The second direction does not require migrating every native class at once.
A bounded trial could share one value type or one query contract and measure
compile portability, numerical agreement and runtime cost. A broad replacement
of native ownership with device-specific storage would be a different,
larger framework change.

Shared environment construction
-------------------------------

**Current implementation.** The application constructs one native C8 scene.
Export preserves its ordered topology and supported model parameters; kernels
query the resident representation. CPU tracking and fallback retain access to
the native scene. Thus scene construction is already shared, while some
evaluation implementations are separate.

The sharing boundary is precise. Ordered boundary selection uses
`NextEnvironmentBoundaryBody.inl`_ through native and numerical access adapters.
Point location follows equivalent ordering in a separate numerical
implementation; density, grammage and intersection calculations are not all
a single shared implementation. The distinction matters when reviewing a
claim that the two paths use the same environment logic.

For ice, firn or another scene made from supported objects, reuse the native
construction and export interfaces. A new density law or shape requires
additional export and evaluation support, not a second independently specified
scene. Material tables, scene admission and radio propagation remain separate
capabilities. Exporting a new geometry does not establish that every physics
or radio model can run there. See :doc:`../environment/extension` and
:doc:`../development/extensions`.

**Candidate maintenance direction.** Share numerical model functions beneath
native and device adapters where their domains and conventions agree. Keep
host object ownership and resident array ownership separate. This could
reduce duplicated formulas without requiring the device to traverse native
smart-pointer or polymorphic object graphs. It would still need explicit
agreements on units, rounding, unsupported domains and error reporting.

**Review evidence.** Test export completeness, coordinate conversion, node
ordering, boundary identity and distance, density/grammage inverses, and
particle round trips. Include changed-scene rejection and runtime ownership:
mutating native objects does not refresh resident data, and ordinary runtime
ownership does not permit arbitrary destruction and reinitialization in one
process. See :ref:`backend-runtime-owner` and :ref:`environment-validation`.

Questions for interface review
------------------------------

These questions identify decisions still needed for an upstream interface;
they do not imply that the current implementation has already adopted the
corresponding change.

1. Should the extension point be a cascade factory, an execution policy, or
   the current adapter? Which framework layer owns stack routing and event
   completion under that choice?
2. Which preparation services should remain independent of execution
   selection? In particular, should writer conventions, photonuclear adapter
   construction and diagnostics be configured through separate interfaces?
3. Should model and writer access come from explicit typed references or
   process-sequence lookup? The current implementation uses both; sequence
   extraction must keep supported types and missing/ambiguous-process behavior
   explicit. See `SequenceModels.inl`_ and `SequenceOutputs.inl`_.
4. Which context and value types should be framework-owned, and which should
   remain implementation-specific? Specify borrowed lifetimes and optional
   dependencies before choosing the containing namespace or directory.
5. What is the supported model-export contract? Decide who maintains export
   adapters and numerical queries when a native shape or material model is
   added, and what happens when only the native path supports it.
6. Which shared numerical function is suitable for the first bounded trial?
   Agree its reference tests, supported backends and acceptance criteria before
   replacing separate implementations.

Implementation references
-------------------------

The links below identify current contracts; the alternatives above are design
options rather than additional functions hidden elsewhere in the repository.

.. _d5297c43: https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397
.. _CascadeExtension.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeExtension.inl
.. _CascadeDriver.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeDriver.inl
.. _CascadeContext.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/cascade/CascadeContext.hpp
.. _EnvironmentSnapshot.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/environment/EnvironmentSnapshot.hpp
.. _RouterParticleConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/data/RouterParticleConversion.inl
.. _NextEnvironmentBoundaryBody.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl
.. _SequenceModels.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/SequenceModels.inl
.. _SequenceOutputs.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/SequenceOutputs.inl
