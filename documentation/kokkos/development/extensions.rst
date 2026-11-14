Extending models and execution paths
====================================

Start an extension at the interface that owns its meaning: a native model,
a numerical query, a final-state implementation, a radio propagator or an
execution-space adapter. Keep application code responsible for constructing
the native C8 scene and configuring the run. Scheduler queues, table layouts
and process-specific device dispatch belong below that application interface.

This page maps recommended changes against the implementation at
`d5297c43 <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
An extension checklist does not imply that the new model or backend already
works. The existing support limits are in :doc:`../environment/extension`,
:doc:`../physics-data/contracts`, :doc:`../radio/propagation` and
:doc:`../backends`.

Changing prepared inputs and runtime lifetime
---------------------------------------------

Prepare changed scene, calculator or radio inputs for a new run. For an
initialized ordinary ``EmRunSession``, use a fresh process: its backend owns
Kokkos initialization, a second ordinary owner is rejected while it is live,
and releasing the final owner finalizes Kokkos. Initialization cannot be
restarted after finalization. Destroying a session and constructing another
in the same process is therefore not a supported reconfiguration recipe.
A generic ``Runtime`` or external ``Kokkos::ScopeGuard`` does not remove
that restriction; ``BackendRuntime`` does not adopt those owners.

The explicit CUDA/OpenMP cooperative lease is a separate lifetime contract.
Any extension implementing in-process backend replacement through that
contract must retain a compatible live lease, complete outstanding work and
release all dependent views before the final lease. This is not a general
session-reconfiguration facility. See :ref:`backend-runtime-owner` and
`BackendRuntime.cpp`_.

Environment models, shapes and materials
----------------------------------------

The current preparation chain is
``prepare_cascade`` → ``prepare_environment_snapshot(environment, coordinates)``
→ ``EnvironmentSnapshot`` → ``export_shape`` / ``export_model``.
It produces ordered host topology, shape, model and composition arrays;
``bind_environment_data`` borrows those arrays and
``EnvironmentStorage<TExecutionSpace>`` uploads them for queries.
The native tree remains the source for host tracking and particle conversion.
See `CascadePreparation.inl`_ and `EnvironmentPreparation.inl`_.

For a new combination of already supported objects, retain this generic
construction path. For a new representation, review the following chain.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Change
     - Host export and records to extend
     - Numerical operations and downstream impact
   * - Density law
     - ``IDensityModelExport::exportDensity()`` returns
       ``DensityModelParameters``; ``export_model`` converts these to
       ``EnvironmentModel`` in common coordinates.
     - Extend ``density(model, positionM)``,
       ``grammage(model, positionM, direction, lengthM)`` and
       ``arclength(model, positionM, direction, grammageGPerCm2)`` together.
       Outputs are g/cm³, g/cm² and metres respectively. Preserve inverse
       domains and failure behavior, not just pointwise density.
   * - Refractivity law
     - ``IRefractivityModelExport::exportRefractivity()`` returns
       ``RefractivityModelParameters``; extend the corresponding
       ``EnvironmentModel`` kind and conversion.
     - ``refractive_index(model, positionM)`` returns dimensionless index.
       Review any density coupling and the separate radio export/propagation
       path; an environment query does not install a radio propagator.
   * - Shape
     - Extend ``EnvironmentShape`` and the exact-type conversion in
       ``export_shape(volume, coordinates)``.
     - Supply ``contains``, ``straight_intersect`` and the required
       ``CurvedBoundaryAccess::intersectShape`` behavior. Retain ordered node
       traversal and boundary selection; a new shape alone need not change
       either algorithm.
   * - Material/composition
     - ``export_model`` exports the native medium identifier and ordered
       nuclear codes/number fractions into ``EnvironmentComponent`` records.
     - Review scene policy, native calculator coverage, auxiliary data and
       CPU completion's composition lookup together. Exporting a medium
       identifier does not supply its interaction tables.

The main edit locations are `ModelConversion.inl`_, `ShapeConversion.inl`_,
`ModelEvaluation.inl`_, `ModelGrammage.inl`_, `EnvironmentSearch.inl`_,
`StraightIntersections.inl`_ and `CurvedBoundaryAccess.inl`_. The existing
boundary selector uses `NextEnvironmentBoundaryBody.inl`_ beneath native and
numerical adapters. Keep shared selection semantics there; do not add a
second geometry search in the application to accommodate a model.

Current restrictions still apply: arbitrary meshes are not exported shape
kinds, and a box has no genuinely curved intersection implementation in this
path. Cascade policies accept air, homogeneous ice or exponential firn under
their documented constraints. The native table exporter and auxiliary builder
also restrict the material physics they accept. General mixed-material
transport therefore needs work beyond relaxing a scene-policy check.

When adding fields, update ``make_environment_snapshot_key`` in
`EnvironmentSnapshotKey.inl`_: it explicitly enumerates parameters rather
than hashing whole records. Its result is also used by
``environment_snapshot_fingerprint``. Review storage transfer, byte accounting
and frontier compatibility when adding arrays or changing record meaning.
Rebuild the snapshot after changing native objects and prepare the new scene
under the runtime-lifetime rules above; retaining an old snapshot does not
refresh its parameters.

Extend conversion, evaluation/grammage and boundary tests in
``tests/kokkos/environment/``, including their device variants. Add transformed
coordinates, surface/tangent cases, inverse round trips and unsupported-input
cases. For changed fields, add a session-reuse rejection case in
`Session.cpp`_. The complete existing test map is in
:ref:`environment-validation`.

Physics processes, native tables and CPU completion
---------------------------------------------------

The current table chain is native calculator views →
``export_proposal_native_tables(interactions, continuous)`` →
``ProposalNativeTableSet`` → ``validate_proposal_native_requirements`` →
``NativeTableStorage`` → ``ProposalNativeDeviceView`` in ``PhysicsContext``.
The exporter copies the configured native interpolation state; it does not
sample a replacement model. Its ``classify_kinematic_model`` accepts explicit
process/PID/parametrization tuples and rejects unknown ones.

For a new process, first decide whether the backend can select its interaction,
complete its final state, or both. These are separate capabilities:

1. Extend the configured native model and export contract. In
   `NativeTableExporter.cpp`_, map the exact parametrization and its loss
   transform. If new records are required, change `ProposalNativeData.hpp`_,
   the owning table, validation/hash encoding and storage together. Update
   ``validate_proposal_native_requirements`` when the new routed species or
   table domain changes the required coverage.
2. Implement the corresponding numerical queries in
   `ProposalNativeQueries.inl`_. Preserve native units, total-rate distance
   sampling, ordered channel selection and stopping-power summation order.
   A cumulative-rate table is not automatically a normalized final-state CDF.
3. Register final-state routing in `ProcessCapabilities.inl`_. A device path
   must extend ``classify_photon_final_state`` or
   ``classify_lepton_final_state`` and its materializer: classification returns
   retained-child counts and outcome flags; materialization writes particles
   and records into assigned slots. Preserve random identities, ancestry,
   energy accounting, suppression and thinning conventions described in
   :doc:`../transport/interactions`.
4. For a declared CPU completion, retain the reached vertex, selected process,
   component and random-selection information in the fallback record.
   Review ``ProposalCpuFallbackHandler::canHandle`` and ``handle`` in
   `ProposalCpuFallbackHandler.inl`_, the native ``doSpecifiedInteraction``
   interface and the router's acceptance policy. The handler executes native
   secondary processing; it must not resample the physical distance already
   traversed. An unknown process or a numerical error must not become an
   implicitly accepted CPU interaction.

Direct electron–positron pair production by ``e±`` and ``μ±`` projectiles
illustrates the current boundary: its rate and loss selection are available,
but the missing native energy-asymmetry CDF keeps complete final states on
CPU. Adding a device completion requires the missing physical information
and validation, not only changing its capability entry.

Representation changes must keep ``ProposalNativeTableFormatVersion``
(currently 7), ``validate_proposal_native_table`` and
``calculate_proposal_native_hash`` consistent. The hash covers canonical
physics content and excludes cache-hit diagnostics. For auxiliary changes,
review both ``ProposalNativeAuxFormatVersion`` (currently 4) and
``ProposalNativeAuxAlgorithmVersion`` plus key/payload generation in
`ProposalNativeAux.cpp`_. Change the applicable identity when old cached data
would no longer represent the new calculation; do not treat deleting one
developer's cache as a compatibility policy.

``EmRunSession::beginProposalNative`` exports tables and auxiliary data only
on its first initialization. Later calls compare environment/geometry keys
and reset event state; they do not re-export changed calculators. Changed
physics needs a newly initialized backend, with a fresh process for ordinary
session ownership as described above. See `EmRunSession.cpp`_.

Extend native export/storage/query tests, discrete/continuous contracts,
final-state tests and CPU fallback tests according to the changed stage.
Include cold/warm-cache identity, invalid-format/hash rejection and selected
vertex replay. Real-table tests need the export-enabled build. Existing
touchpoints are mapped in :ref:`physics-data-validation` and
:doc:`../transport`; a new physical model also needs independent physics
validation and representative shower comparisons.

Radio propagation
-----------------

The ordinary path exports a flat-atmosphere profile or homogeneous index,
then ``propagate(propagation, source, observer)`` returns ``RadioSignalPath``
for CoREAS/ZHS projection. To extend that path, change the propagation record,
host configuration/export, uploaded storage and `RadioProjection.inl`_
together. Keep the emission formalism, track weighting, time bins and output
units explicit; changing ray propagation alone does not justify changing
those contracts.

For a distinct propagation engine, `IRadioBackend.hpp`_ supplies a host
submission boundary: ``enqueueRadioTracks(span)`` must copy borrowed
``LeptonTransportRecord`` inputs before returning;
``downloadRadioWaveforms()`` must finish pending work, including a partial
batch, and return ``RadioWaveforms``. Its general interface permits an
implementation to own different optical data, but supplies no general
geometry solver itself. Keep kernel dependencies and model preparation
behind the adapter.

Firn already has separate ``IGeometricalOpticsAccumulator`` and
``IFullWaveAccumulator`` interfaces. A new optical profile needs an explicit
device representation and solver; the current GO solver accepts exponential
parameters. A changed full-wave table contract must update both producer and
consumer: `FullWaveAccumulator.cpp`_ checks schema 2, profile fingerprint,
interpolation convention, NPZ checksum and array coverage. These interfaces
do not make arbitrary transport interfaces or the host-only hybrid model
available on devices.

Review configuration reuse and merging as well as projection. Ordinary
``RadioConfig`` is prepared only before the backend exists, and
``same_radio_physics`` in `CooperativeRadioMerge.inl`_ explicitly compares
propagation and observer fields. Extend that comparison for new physical
parameters and prepare a new configured run under the lifetime rules above
when those parameters change.
Add projection/reference cases, accumulator tests, final-partial-batch replay
and merge-mismatch cases; use the firn accumulator tests for its separate
interfaces. See :doc:`../radio` for the existing test map.

Execution backends
------------------

An additional execution space should reuse the numerical stage interfaces.
The current chain is build selection →
``resolve_execution_backend(requested)`` → ``EmBackend`` factory →
``BackendInstanceImplementation<TExecutionSpace>``. The latter owns runtime,
execution instance, tables, environment and event storage. A factory receives
``RuntimeConfig`` and returns a ``BackendInstance``; it does not define a
different physical model.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Source to review
     - Recommended extension work
   * - `BackendRuntimeSelection.cmake`_, `BackendRuntimeTargets.cmake`_
     - Recognize the supported Kokkos device combination, compile the runtime
       with the required compiler and export its target/dependencies.
   * - `BackendSelection.cpp`_, `BackendRuntime.cpp`_
     - Resolve the requested backend; implement initialization, lifetime,
       device metadata and relevant resource queries.
   * - `BackendApplicationTargets.cmake`_, `EmBackend.cpp`_,
       `CudaBackendInstance.cpp`_
     - Register and dispatch a narrow factory for the new execution space.
       The CUDA file is an existing example of that boundary.
   * - `BackendInstanceImplementation.inl`_, `TuningCache.cpp`_
     - Review memory budgeting, launch policies and tuning identity. Keep
       physical configuration in the existing model/session interfaces.

Audit native-operation branches such as scans and completion waits, memory
accessibility, raw-view strides and execution-instance ordering. Retain the
portable implementation where applicable. Register kernel-bearing tests with
the correct device compiler, and exercise runtime lifetime, transfers, scans,
resident execution and installed consumers on the intended backend. See
:doc:`../backends/device-code` for these contracts. Factory compilation alone
does not establish hardware support, cooperative execution or coordinator
support; each has separate branches and tests.

.. _CascadePreparation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadePreparation.inl
.. _EnvironmentPreparation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentPreparation.inl
.. _ModelConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ModelConversion.inl
.. _ShapeConversion.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ShapeConversion.inl
.. _ModelEvaluation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ModelEvaluation.inl
.. _ModelGrammage.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/ModelGrammage.inl
.. _EnvironmentSearch.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentSearch.inl
.. _StraightIntersections.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/StraightIntersections.inl
.. _CurvedBoundaryAccess.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/CurvedBoundaryAccess.inl
.. _NextEnvironmentBoundaryBody.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl
.. _EnvironmentSnapshotKey.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/environment/EnvironmentSnapshotKey.inl
.. _Session.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/session/Session.cpp
.. _NativeTableExporter.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/NativeTableExporter.cpp
.. _ProposalNativeData.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ProposalNativeData.hpp
.. _ProposalNativeQueries.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/tables/ProposalNativeQueries.inl
.. _ProcessCapabilities.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/ProcessCapabilities.inl
.. _ProposalCpuFallbackHandler.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/proposal/ProposalCpuFallbackHandler.inl
.. _ProposalNativeAux.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/proposal/ProposalNativeAux.cpp
.. _EmRunSession.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp
.. _RadioProjection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/algorithms/RadioProjection.inl
.. _IRadioBackend.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/IRadioBackend.hpp
.. _FullWaveAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/radio/firn/FullWaveAccumulator.cpp
.. _CooperativeRadioMerge.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/CooperativeRadioMerge.inl
.. _BackendRuntimeSelection.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendRuntimeSelection.cmake
.. _BackendRuntimeTargets.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendRuntimeTargets.cmake
.. _BackendSelection.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/BackendSelection.cpp
.. _BackendRuntime.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/BackendRuntime.cpp
.. _BackendApplicationTargets.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendApplicationTargets.cmake
.. _EmBackend.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmBackend.cpp
.. _CudaBackendInstance.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/CudaBackendInstance.cpp
.. _BackendInstanceImplementation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl
.. _TuningCache.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/TuningCache.cpp
