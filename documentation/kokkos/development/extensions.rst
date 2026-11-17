Extending environments and physics capabilities
===============================================

A new scenario can often be built from existing C8 objects without adding a
new execution backend. Start by checking whether the geometry, material law
and physics model are already supported by the exporters and numerical
functions. Extend those components only where a required capability is missing.

This page gives development steps and code locations. Supported domains and
current limitations remain documented in :doc:`../environment/extension`,
:doc:`../physics-data/contracts` and :doc:`../radio/propagation`.

Adding an environment
---------------------

The application constructs the native C8 environment tree. The existing
preparation path then performs:

.. code-block:: text

   C8 environment and coordinate system
       -> prepare_environment_snapshot
       -> EnvironmentSnapshot: nodes, shapes, models and composition
       -> EnvironmentStorage: device-accessible arrays
       -> density, grammage, node and boundary queries

For a different arrangement of supported shapes and models, reuse that path.
Do not add a second environment tree or a scene-specific particle scheduler.
For a new model or shape, update the following parts together.

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Capability
     - Files and operations to extend
     - Required comparison
   * - Density law
     - The native density export, ``EnvironmentModel``, `ModelConversion.inl`_,
       ``density`` in `ModelEvaluation.inl`_, and ``grammage`` / ``arclength``
       in `ModelGrammage.inl`_.
     - Point density, integrated grammage and inverse path length, including
       transformed coordinates and the model's valid domain.
   * - Geometric shape
     - ``EnvironmentShape``, `ShapeConversion.inl`_, containment,
       `StraightIntersections.inl`_ and `CurvedBoundaryAccess.inl`_.
     - Containment and first boundary crossing; include tangent, surface and
       nested/excluded-node cases.
   * - Material composition
     - Native medium/composition export, scene preparation and native physics
       calculator coverage.
     - Correct material identity, component fractions, available interaction
       tables and CPU completion for that material.
   * - Refractivity or propagation
     - Native refractivity export and numerical index evaluation, plus the
       separate radio configuration and propagation model.
     - Travel time, path geometry and waveform response. A density export does
       not automatically provide the corresponding radio model.

The native density adapter exposes ``IDensityModelExport::exportDensity()``.
``export_model`` converts its parameters into the numerical record. The
queries return density in g/cm³, grammage in g/cm² and path length in metres;
these units are stated in :doc:`../environment/queries`.

**Existing example: exponential firn.** ``corsika/media/ExponentialFirn.hpp``
provides the native medium. Its parameters are exported through
``ModelConversion.inl`` and evaluated by the shared environment queries and
``environment/firn`` implementation. ``application/cascade/firn`` handles the
scene's preparation constraints; ``radio/firn`` handles propagation. A new
firn profile should follow these responsibilities, rather than copying the
whole cascade implementation. Current firn support does not imply arbitrary
material mixtures or arbitrary refractivity profiles are supported.

Keep native parent, child and excluded-node relationships in the snapshot.
Add new parameters to ``make_environment_snapshot_key`` in
`EnvironmentSnapshotKey.inl`_ so a reused session cannot silently accept a
changed scene. Environment tests live in ``tests/kokkos/environment``; start
with ``testModelConversion.cpp``, ``testModelEvaluation.cpp`` and the
appropriate grammage, boundary and device tests. ``testFirnEnvironment.cpp``
shows the existing firn-specific checks.

Adding a stochastic reaction
----------------------------

Three capabilities must be connected: selecting the reaction, producing its
final state, and inserting the resulting particles. Support for the reaction
rate alone does not mean the final state can execute on the GPU.

1. **Prepare the physical inputs.** Configure the native model and extend
   ``export_proposal_native_tables`` in `NativeTableExporter.cpp`_ if its
   parametrization or table layout is new. Preserve the model identity, units
   and native sampling information. Update the table validator and storage if
   the exported representation changes.
2. **Connect rate and sampling queries.** Extend `ProposalNativeQueries.inl`_
   and the selection path where needed. Test with the same tables, energy and
   random inputs as the reference. Preserve process/component ordering when
   the change is intended to reproduce existing selection.
3. **Choose final-state execution.** Update `ProcessCapabilities.inl`_ and the
   corresponding photon/lepton classifier and materializer for a device
   implementation. Alternatively, use the existing CPU completion interface
   in `ProposalCpuFallbackHandler.inl`_. That handler receives the selected
   vertex and process; it must not sample the already traversed distance again.
4. **Check particle and energy accounting.** Verify the outgoing species,
   energies, directions, weights and identities, including cut and thinning
   behavior. Exercise both the ordinary result and any CPU completion path,
   then run an appropriate shower comparison.

Direct electron–positron pair production by electron/positron and muon
projectiles illustrates this separation. The current backend has rate and
loss selection data, but lacks the native energy-asymmetry CDF needed for a
complete device final state. Those final states are completed on CPU. Changing
only the capability flag would omit the missing calculation.

Use the export/storage/query tests in ``tests/kokkos/em/tables`` and the
selection, final-state and fallback tests described in
:doc:`../transport/interactions`. A new physical model also requires a physical
reference; passing an old-code comparison alone is insufficient.

Extending radio or execution
----------------------------

For radio changes, distinguish the source calculation from propagation and
accumulation. Extend the relevant CoREAS/ZHS projection or propagation
implementation, then test the returned path, time bins and waveform. The
host ``IRadioBackend`` interface accepts borrowed track records through
``enqueueRadioTracks``; implementations copy them before returning.
``downloadRadioWaveforms`` completes pending work, including a partial batch.
Firn also has dedicated geometrical-optics and full-wave accumulator interfaces.
Their supported inputs and tests are in :doc:`../radio`.

For a new execution backend, reuse ``BackendInstanceImplementation`` and the
numerical stage functions. Extend build selection, runtime initialization,
resource queries and the backend factory. For an existing backend's performance
change, work on launch policies, batching or storage as appropriate. Neither
case should require another copy of the physical model. Follow
:doc:`../backends` and :doc:`../validation/performance` for device compilation,
ordering and measurement requirements.

Preparing changed inputs
------------------------

Environment snapshots, physics tables and radio configuration are prepared
before event execution. A reused ``EmRunSession`` resets event state; it does
not reload changed calculators or transport cuts. Build changed physical
inputs for a new run.

For the ordinary production session, use a fresh process when those inputs
change. Its backend owns Kokkos initialization, and Kokkos cannot be initialized
again after finalization in the same process. The explicit cooperative lease
has different ownership rules and is not a general reconfiguration interface.
See :ref:`backend-runtime-owner` before changing runtime ownership.

If an extension changes stored parameters, update the corresponding cache key
or format, storage transfer and reuse comparisons. The inventory in
:doc:`formats` identifies those mechanisms; it need not be changed for a
feature that reuses the existing records unchanged.

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
