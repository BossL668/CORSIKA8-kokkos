Physics tables and transport stages
===================================

These declarations connect configured native models to numerical queries and
individual particle stages. They are generated from the public headers at the
baseline recorded in :doc:`../api`. Host preparation owns and validates data;
numerical functions borrow accessible arrays and return records or status.
They do not own the surrounding scheduler.

For the complete preparation chain, supported processes and array layout, see
:doc:`../physics-data`. For the ordering of sampling, propagation, scattering,
vertex selection and final-state production, see :doc:`../transport`.

Native export and ownership
---------------------------

The exporter receives views of native calculators already configured by C8.
The resulting host table owns the copied interpolation state. Uploading it
creates a separate execution-space owner; the returned numerical view borrows
that owner's arrays. A table-view name does not imply that its pointers are
accessible on every device.

Sources: `NativeTableExporter.hpp`_, `ProposalNativeTable.hpp`_ and
`NativeTableStorage.hpp`_. Cache identity and runtime lifetime are described
in :doc:`../physics-data/preparation` and :doc:`../physics-data/storage`.

.. doxygenfunction:: corsika::kokkos::export_proposal_native_tables

.. doxygenstruct:: corsika::kokkos::ProposalNativeTableSet
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::validate_proposal_native_table

.. doxygenclass:: corsika::kokkos::NativeTableStorage
   :members: projectedDeviceBytes, initialize, reset, isInitialized, getDeviceBytes, getSourceContentHash, getDeviceView
   :undoc-members:

Native query results and units
------------------------------

The following queries use total energies in MeV and grammage in g/cm².
Rates are per g/cm²; the loss-fraction result is dimensionless. These units
are distinct from particle-state total energies in GeV. Callers inspect the
returned status before using the value. Query records also retain iteration
diagnostics; they do not own table storage.

The cumulative-rate query takes the table's transformed loss coordinate.
The inverse query instead takes a quantile and returns a physical loss or
secondary-energy fraction according to its ``photon_split`` argument. The
independent total-rate spline is used for distance sampling; ordered channel
selection has its own rate-summation query. See
:doc:`../physics-data/storage` for those distinctions.

Sources: `ProposalNativeData.hpp`_ and `ProposalNativeQueries.hpp`_. The
declarations below select the principal query boundaries; the header also
provides individual-column, stopping-power, range and selection-total queries.

.. doxygenstruct:: corsika::kokkos::NativeQueryResult
   :members:
   :undoc-members:

.. doxygenfunction:: corsika::kokkos::query_proposal_native_total_rate

.. doxygenfunction:: corsika::kokkos::query_proposal_native_cumulative_rate

.. doxygenfunction:: corsika::kokkos::query_proposal_native_loss_fraction

.. doxygenfunction:: corsika::kokkos::query_proposal_native_energy_after_continuous_loss

Selection and propagation
-------------------------

Selection produces a candidate interaction record without advancing the
particle. Transport then resolves the competing limits and returns the
completed step or an explicit failure/completion request. The charged-lepton
transport function does not itself execute multiple scattering or a discrete
final state. Its ``applyMoliere`` argument also controls the zero-length-cut
status consumed by the later scattering stage.

These functions are numerical entry points, not host queue-submission calls.
The caller supplies exclusive output slots and keeps all borrowed table and
environment data alive in accessible memory. The meaning of partial records
and return flags is part of the interface; a fallback record does not execute
the CPU handler. See :doc:`../transport/selection`,
:doc:`../transport/lepton-step` and :doc:`../execution/batches`.

Sources: `InteractionSelection.hpp`_, `LeptonTransportStep.hpp`_ and
`PhotonTransportStep.hpp`_.

.. doxygenfunction:: corsika::kokkos::select_discrete_interaction

.. doxygenfunction:: corsika::kokkos::transport_lepton

.. doxygenfunction:: corsika::kokkos::transport_photon

Final-state classification and materialization are separate public stage
interfaces in `LeptonFinalStateStep.hpp`_ and `PhotonFinalStateStep.hpp`_.
Their counting, exclusive-slot assignment, CPU completion and energy-accounting
contracts are documented in :doc:`../transport/interactions`.

.. _NativeTableExporter.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/proposal/NativeTableExporter.hpp
.. _ProposalNativeTable.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ProposalNativeTable.hpp
.. _ProposalNativeData.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ProposalNativeData.hpp
.. _NativeTableStorage.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/NativeTableStorage.hpp
.. _ProposalNativeQueries.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ProposalNativeQueries.hpp
.. _InteractionSelection.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/InteractionSelection.hpp
.. _LeptonTransportStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/LeptonTransportStep.hpp
.. _PhotonTransportStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/PhotonTransportStep.hpp
.. _LeptonFinalStateStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/LeptonFinalStateStep.hpp
.. _PhotonFinalStateStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/PhotonFinalStateStep.hpp
