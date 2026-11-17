.. _execution-granularity-and-algorithm-ownership:

Maintaining execution and reproducibility
=========================================

Execution changes affect how particles reach a calculation, how their results
are retained and when native processing resumes. Maintain those transitions
alongside the kernels themselves. This page describes the responsibilities at
`revision d5297c43`_ and the checks needed when changing them.

Development procedure
----------------------

1. Locate the change in the resident kernel pipeline, host router or process
   coordinator. Keep numerical algorithms unchanged when evaluating only an
   execution change.
2. Trace each input to its continuation, termination or CPU-completion
   result. Preserve particle identity, weights, random state and accumulated
   outputs across those transitions.
3. State whether the change should preserve random histories. Reuse and
   partitioning can change stream progression even with the same seed labels.
4. Run the relevant component and event checks, then measure both the changed
   stage and complete events under the same physical conditions.

.. _decision-shared-calculations:

Numerical algorithms and execution code
---------------------------------------

CUDA and HIP factories instantiate the same
``BackendInstanceImplementation<TExecutionSpace>`` behind `IEmBackend`_. A
launch-policy change can therefore remain in execution code while both
backends continue to call the same physical calculation. Review
backend-specific branches for scans, waits and resource queries separately;
see :doc:`../backends/device-code`.

Sharing with native C8 is more selective. Native and numerical boundary
selection already include `NextEnvironmentBoundaryBody.inl`_ through different
access adapters. PROPOSAL preparation instead uses `NativeTableExporter.cpp`_
to export interpolation data, with queries implemented separately in
`ProposalNativeQueries.inl`_. An upstream change to those calculations must be
reviewed at the exporter and query boundary as well as in the native model.

:doc:`../development/shared-algorithms` gives a concrete procedure for reducing
this duplication. Keep an independent reference when extracting a common
function; changing the implementation and its test oracle to call the same
new helper can remove the comparison needed to detect a defect.

.. _decision-execution-policy:

Resident batches and host routing
---------------------------------

On the ordinary single-endpoint path, ``EmRouter`` separates photons from
charged leptons and submits bounded inputs. `LeptonFront.inl`_ and
`PhotonFront.inl`_ launch the relevant selection, transport and final-state
work. Stable scans assign output slots; photon final-state work uses the
compacted list of reached interaction vertices. Resident execution can advance
several wavefronts and retain particles between host checkpoints.

When changing this pipeline, review three boundaries together:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Boundary
     - Behavior to preserve
     - Code and tests
   * - Input submission
     - A complete state and random key for each submitted particle; accessible
       immutable tables and environment data.
     - :doc:`../particles/conversion`, :doc:`../execution/batches`
   * - Device result queues
     - Each continuation, secondary, termination and returned request is
       accounted for once, including capacity-limited returns.
     - :doc:`../particles/storage`, :doc:`../execution/batches`
   * - Host completion
     - The returned state resumes the correct native process and contributes
       to the same output accounting.
     - :doc:`../execution/hybrid-loop`, :doc:`../transport/interactions`

Particle identity belongs to the record and random key, not a permanently
assigned thread. Launch geometry may change without redefining that identity.
For performance work, compare sparse event tails as well as large batches:
fewer launches can still increase register use, divergence or queue costs.
Measure the complete event before interpreting a kernel improvement as a
shower speedup.

The existing independent-subshower path and ``IndependentSubshowerPump`` in
`CooperativeBackend.cpp`_ coordinate host execution endpoints. They are not a
persistent task queue inside a GPU. A future GPU task scheduler would need
its own secondary ownership, termination and overflow design; it should be
evaluated as an execution change rather than added inside a physics function.

Native completion and process coverage
--------------------------------------

`ProcessCapabilities.inl`_ declares final-state support. For selected CPU
interactions, the returned record retains the reached vertex and process
information. `ProposalCpuFallbackHandler`_ completes accepted requests through
native C8 processing. Small-batch scalar work, forced decay and memory-spill
returns have different meanings and should retain their own handling.

Direct electron–positron pair production by electron/positron and muon
projectiles uses CPU completion because the native energy-asymmetry CDF is
unavailable. A rate table alone does not provide the missing final-state
distribution. Keeping these interactions on CPU preserves their contribution
to the simulation.

When adding device support for a reaction, update the capability declaration,
final-state calculation and return handling together. Compare the complete
reaction with its native reference, including surviving and newly produced
particles. When an upstream model changes but device support remains absent,
maintain the native completion path rather than silently removing the process.
:doc:`../development/extensions` describes the extension steps.

Worker reuse and event partitioning
-----------------------------------

`AirShowerCoordinator`_ partitions frontier roots into fixed worker
assignments. Each worker runs a local hybrid cascade; the coordinator verifies
and merges the results. CPU-prefix worker count and GPU worker count are
separate settings. There is no inter-worker work stealing in this path.

Persistent workers reuse model and backend resources across compatible
``kokkos-proposal`` events. GPU initialization occurs on the first non-skipped
request, while model preparation happens earlier. Events are still processed
sequentially through the coordinator. The lifecycle and input/output records
are documented in :doc:`../execution/coordinator`.

A reuse change must distinguish persistent resources from per-event state.
Keep tables, geometry and configuration compatible with the retained backend;
reset or replace the event-specific records and output destinations as defined
by the session interface. Include multiple requests and final shutdown in the
regression test. If a model, scene or cut changes, follow
:doc:`../development/compatibility` rather than assuming a live session can
accept it or that Kokkos can be reinitialized after finalization.

A partitioning change also affects who owns imported particles and merged
outputs. Equal root counts do not guarantee equal work. Workload-based
assignment or dynamic redistribution would require measurements as well as
complete state and output-transfer rules. Review worker formats and merger
behavior through :doc:`../development/formats` when changing those records.

.. _decision-reproducibility-policy:

Random streams and acceptance criteria
--------------------------------------

Reused workers continue native CPU random streams. Device sampling uses the
request seed and particle history. A reused worker is therefore not generally
equivalent to a fresh process started for each seed.

Choose the comparison according to the intended change. A pure interface
refactor should preserve the specified records and lifecycle behavior. A
numerical-method change needs an error analysis against an appropriate
reference. A scheduling or stream-policy change may require a statistical
shower comparison. In every case, process coverage, complete particle-state
transfer and output accounting remain part of correctness.

:doc:`../validation/numerics` explains these comparisons and their references.
For timing, use matched physics, resource allocation, initialization/reuse
policy and output work. Report the complete-event interval alongside the
changed kernel or scheduling phase using :doc:`../validation/performance`.
Wall time, accumulated device activity and allocated device-hours describe
different costs and should retain separate definitions.

.. _revision d5297c43: https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397
.. _IEmBackend: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/IEmBackend.hpp
.. _LeptonFront.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/LeptonFront.inl
.. _PhotonFront.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/PhotonFront.inl
.. _NextEnvironmentBoundaryBody.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl
.. _NativeTableExporter.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/NativeTableExporter.cpp
.. _ProposalNativeQueries.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/tables/ProposalNativeQueries.inl
.. _ProcessCapabilities.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/ProcessCapabilities.inl
.. _ProposalCpuFallbackHandler: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/proposal/ProposalCpuFallbackHandler.inl
.. _AirShowerCoordinator: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp
.. _EmRouter.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/EmRouter.inl
.. _CooperativeBackend.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/CooperativeBackend.cpp
