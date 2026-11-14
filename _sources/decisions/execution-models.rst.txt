Execution granularity and algorithm ownership
=============================================

The integration separates the physical calculation from the mechanism that
schedules it. That separation is incomplete across native C8 and the numerical
backend: some operations share an implementation, while others reproduce the
native algorithm through exported data. Maintenance decisions should identify
which boundary is changing rather than treat all parallelization work as one
choice.

The implemented behavior below refers to `revision d5297c43`_. Alternatives
are discussion options, not implemented features or measured performance
claims. The execution contracts and tests remain the acceptance criteria for
any replacement.

.. _decision-execution-policy:

Batch execution and persistent particle tasks
---------------------------------------------

**Implemented.** On its single-endpoint wavefront path, ``EmRouter`` separates
photons from charged leptons, submits bounded inputs and consumes returned
records. Resident loops can advance
multiple wavefronts before a checkpoint; eligible particles and secondary
queues can remain resident between calls. `LeptonFront.inl`_ and
`PhotonFront.inl`_ launch selection, transport and final-state work, with
stable scans assigning output slots. Photon final-state work is restricted
to a compacted list of reached interaction vertices. This is not a transfer
of every particle to the host after every physical step.

The host `IEmBackend`_ interface is independent of the execution-space
implementation. CUDA and HIP factories instantiate the same
``BackendInstanceImplementation<TExecutionSpace>``. Particle identity belongs
to the state and random key, not to a permanently assigned hardware thread.
The actual call and return paths are documented in :doc:`../execution/batches`.

The current design provides explicit stage ordering, bounded storage and
well-defined checkpoints for native completion and output. Its costs include
kernel launches, scans, queue management and host observation at checkpoints.
The relative cost depends on batch population and the physical workload;
the existence of these operations does not establish that they dominate a
complete shower.

.. list-table::
   :header-rows: 1
   :widths: 25 36 39

   * - Discussion option
     - Potential benefit to evaluate
     - Additional obligations
   * - Retain wavefront execution; tune policies or combine compatible stages
     - Reduce launch or intermediate-storage work while preserving the current
       scheduling boundary.
     - Measure register/local-memory use, queue costs and complete-shower time.
       A combined kernel can execute more divergent work or reduce occupancy;
       compare it with the separated path on each intended device.
   * - Persistent per-particle or per-subshower tasks
     - Continue several steps locally and obtain another task from a device
       queue without returning through every existing stage boundary.
     - Specify queue synchronization, termination, overflow, secondary
       ownership, CPU completion and output ordering. Measure branch divergence,
       register use and load imbalance; reduced launch count alone is not a
       performance result.

**Open decision.** Should a future scheduler retain the existing wavefront
interface, or introduce a separate task-execution contract? A prototype should
keep physical inputs fixed, account for every continuation and output once,
and compare both representative large batches and sparse event tails.
Persistent tasks inside a device are distinct from the existing persistent
*process workers* described below. Neither approach assigns a scientific
meaning to CUDA blocks, HIP workgroups or host thread identities.

There is already an opt-in independent-subshower path in the host interface.
``EmRouter`` checks it before the ordinary photon/lepton split, and the
CUDA/OpenMP cooperative implementation uses ``IndependentSubshowerPump`` to
coordinate its endpoints. This existing host CPU/GPU cooperation does not
implement the proposed persistent task queue inside one GPU. See
`EmRouter.inl`_ and `CooperativeBackend.cpp`_.

.. _decision-shared-calculations:

Shared numerical implementations
--------------------------------

**Implemented.** Backend factories share numerical stage interfaces, but native
C8 and the Kokkos path do not universally call one implementation. A concrete
shared example is `NextEnvironmentBoundaryBody.inl`_: native and numerical
wrappers supply different access adapters around the same ordered boundary
selection body. It preserves selection logic without exposing native nodes,
units and ownership to device compilation.

The PROPOSAL path has a different boundary. `NativeTableExporter.cpp`_ copies
configured interpolation data, while `ProposalNativeQueries.inl`_ implements
queries against the exported representation. Export is data reuse, not a
runtime call to PROPOSAL inside a device kernel. Final-state capabilities also
remain separate from the availability of rate and loss tables; see
:doc:`../physics-data/contracts` and :doc:`../transport/interactions`.

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Maintenance option
     - Boundary
     - Consequence
   * - Share an algorithm body through access adapters
     - Native and numerical representations provide the operations required
       by one algorithm, as in boundary selection.
     - Both adapters need compatible units, ordering, domains and failure
       semantics. Device compilation must reach only supported operations.
       Changes require tests through both representations.
   * - Extract a device-callable numerical core
     - Host model objects prepare immutable parameters; CPU and device paths
       call a common calculation with explicit state and results.
     - Requires agreement on ownership, parameter representation, math
       dependencies and error reporting. An external library's core needs its
       own maintainer agreement and portability work, not merely a local
       function annotation.
   * - Retain a native reference and separate numerical implementation
     - The native package remains authoritative; the backend exports its
       data and implements the specified operations.
     - Preserves the present integration boundary but requires explicit
       dependency/version checks and regression coverage for every duplicated
       calculation. A shared physical model does not imply identical rounding.

**Open decision.** Which algorithms should become shared C8 infrastructure,
and which should remain adapters to independently maintained dependencies?
Start with a bounded operation whose inputs and result can be specified,
rather than move an entire stateful model object into a kernel. Device-callable
helpers need accessible data, a supported call graph and explicit ownership;
``inline`` or ``KOKKOS_INLINE_FUNCTION`` alone establishes none of those
contracts. :doc:`../backends/device-code` describes the existing requirements.

A shared implementation also needs an independent reference. Moving the test's
oracle to the same new helper can remove the comparison that detects a defect.
Retain suitable native-model, analytic or frozen-reference checks, and document
what each one establishes. Updating a dependency requires reviewing its export
contract, tables and numerical tests together.

Native completion and capability ownership
------------------------------------------

**Implemented.** `ProcessCapabilities.inl`_ declares final-state support.
Selected CPU interactions preserve the reached vertex and process information;
`ProposalCpuFallbackHandler`_ completes accepted requests through native C8
processing. Small-batch scalar expansion, forced decay and memory-spill returns
have different scheduling contracts. They must not be combined into a single
"unsupported physics" category.

Direct electron–positron pair production by electron/positron and muon
projectiles currently uses CPU completion because the native energy-asymmetry
CDF is unavailable. A rate table alone cannot supply that missing distribution.
Other native models may depend on host libraries with their own runtime state.
Keeping their completion on CPU is compatible with a parallel EM backend;
it does not remove the reaction from the simulation.

**Open decision.** Should capability information become a common C8 service,
or remain local to each execution implementation? A common description could
make support discoverable before a run, but must distinguish configured model
availability, selection support, final-state support and supported materials.
The execution implementation still needs to enforce the declared domain and
preserve the selected vertex when returning work. An unknown process or a
numerical failure must not silently acquire a valid completion policy.

A GPU implementation of an additional native model is separate work: audit
its state, random-number source, dependencies and result construction, then
validate the complete reaction. Replacing a CPU return with an approximate
final state solely to reduce transfer time changes the physical calculation.

Worker reuse and event partitioning
-----------------------------------

**Implemented.** `AirShowerCoordinator`_ partitions frontier roots into fixed
worker assignments. Workers execute local hybrid cascades and the coordinator
validates and merges their outputs. There is no inter-worker work stealing.
CPU-prefix worker count and GPU worker count are separate settings. With
persistent workers enabled, multiple ``kokkos-proposal`` events reuse worker
processes and their initialized backend resources. GPU backend initialization
occurs on the first non-skipped request; worker model setup occurs earlier.
Events still execute sequentially through the
coordinator. See :doc:`../execution/coordinator`.

Reuse retains native CPU random streams while device sampling uses the request
seed and particle history. It is therefore not equivalent to launching a fresh
process for each seed. Reuse also assumes compatible prepared inputs: ordinary
sessions cannot be arbitrarily replaced after Kokkos finalization. Model,
scene or cut changes require the lifecycle described in
:doc:`../development/extensions`.

**Open decision.** Should multi-device scheduling continue to use static
frontiers, add workload-based partitioning, or introduce dynamic redistribution?
Static assignments provide explicit ownership and reproducible input artifacts,
but unequal subshower costs can leave workers idle. Redistribution would need
a transfer protocol for particle/random state and accumulated outputs,
termination detection, capacity policy and recovery behavior. It must also
separate particle work from radio work. No universal scaling claim follows
from a balanced root count or a larger worker count.

.. _decision-reproducibility-policy:

Reproducibility and validation
------------------------------

Agree on the intended equivalence before implementation. Interface refactoring
should preserve specified inputs, outputs and lifecycle behavior. Scheduling
or random-policy changes may instead require a declared statistical comparison;
allowing different histories does not waive conservation, process coverage,
queue ownership or completion checks.

For timing, compare the same physical configuration, resource allocation,
initialization/reuse policy and output work. Report the complete event interval
alongside the affected kernels or scheduling phases. Wall time, active device
time and allocated device-hours answer different questions. The methods and
existing checks are in :doc:`../validation/numerics`,
:doc:`../validation/performance` and :doc:`../validation/tests`.

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
