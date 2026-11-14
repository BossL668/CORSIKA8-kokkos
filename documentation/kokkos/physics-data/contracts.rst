Query contracts and model extensions
====================================

Numerical queries read a borrowed native-table view and return explicit
status values. They do not allocate tables, generate random numbers or choose
a different physical model. The transport caller interprets the result and
decides whether to continue, complete an interaction on the CPU or report an
invalid configuration.

Query inputs and results
------------------------

The public contracts are declared in `DiscretePhysicsQueries.hpp`_ and
`ContinuousPhysicsQueries.hpp`_. Energies at this interface are total energies
in MeV, while transport cuts are kinetic energies in MeV. Particle records
use GeV; callers perform the conversion explicitly.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Query
     - Required inputs
     - Result on success
   * - ``query_discrete_rate``
     - Native view, PDG code, process ID, component hash and total energy.
     - Physical column rate in cm²/g.
   * - ``query_discrete_total_rate``
     - Native view, PDG code and total energy.
     - The separate total-rate interpolant used to sample interaction depth.
   * - ``select_rate_column_by_uniform``
     - Native view, PDG code, total energy and a supplied uniform in [0, 1).
     - Selection status, process/component identity, total selection rate,
       residual quantile and a ``selected_`` flag.
   * - ``query_discrete_loss_fraction``
     - Native view, PDG/process/component identity, total energy and quantile.
     - Energy fraction or a query-specific photon-pair split coordinate.
   * - ``query_continuous_dedx`` / ``query_continuous_range``
     - ``ContinuousPhysicsView``, PDG code and total energy.
     - Stopping power in MeV/(g/cm²) or range in g/cm².
   * - ``query_continuous_energy``
     - Continuous view, PDG code and range in g/cm².
     - Total energy in MeV from native range inversion.
   * - ``query_energy_after_continuous_loss``
     - Continuous view, PDG code, initial total energy and grammage in g/cm².
     - Final total energy in MeV, or an explicit cut/failure status.

Most queries return ``TableQueryResult``: ``status_``, ``value_`` and
``reserved_``. In this result type, ``reserved_`` packs saturated Newton and
bisection iteration counts; use ``native_newton_iterations`` and
``native_bisection_iterations`` to read them. This field has a different
meaning from ``ParticleState::reserved_``.

Selection returns ``RateColumnSelectionResult`` instead. ``Success`` alone
does not establish that a column was selected: a successful zero-rate result
has ``selected_ == 0``. Preserve the original column order, strict subtraction
comparison and residual quantile when changing the implementation.

Photon-pair query identities
----------------------------

The two final-state query identifiers do not denote extra reaction channels.
They request a secondary-energy split from an existing photopair column:

* ``PhotonPairFinalStateProcessId`` returns a normalized electron split
  coordinate after native CDF inversion and the existing kinematic encoding.
* ``PhotonMuonPairFinalStateProcessId`` returns :math:`E_{\mu^+}/E_\gamma`.

Ordinary loss queries return the fraction consumed by the interaction. Treating
all these outputs as the same fraction would confuse projectile loss with
secondary sharing. The dispatch and conversion are implemented in
`DiscretePhysicsQueries.inl`_.

Status handling
---------------

`TableQueryResult.hpp`_ defines the public status values. A zero numerical
value is not a substitute for checking ``status_``.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Status group
     - Interpretation
   * - ``Success``
     - A usable result under the query's contract. A physical rate can be zero
       below its process threshold.
   * - ``ParticleNotFound``, ``ColumnNotFound``, ``ContinuousParticleNotFound``
     - The requested identity or continuous utility is absent from the view.
       Absence is distinct from a known channel with zero rate.
   * - Energy, quantile or range out of range
     - The request is outside the corresponding query domain. Read the
       specific status and input; do not silently replace it with an endpoint.
   * - ``InverseCdfUnavailable``
     - Native-query conversion can report unsupported kinematics or an
       unconverged inverse through this status. It does not uniquely identify
       a missing table.
   * - ``NonFiniteInput``, ``InvalidQueryKind``, ``InvalidTableView``
     - Invalid input, dispatch or storage. Some domain checks also use
       ``NonFiniteInput`` for finite but invalid arguments, such as a negative
       grammage.
   * - ``TransportCutReached``
     - The continuous-loss request reaches the available range. This is an
       explicit transport outcome rather than a table-allocation failure.

For continuous step limiting, ``query_continuous_minimum_energy`` applies
``ContinuousCutSafetyFactor`` to the sum of transport mass and kinetic cut.
That limiting energy is not the user particle cut. Use the separate
``query_continuous_transport_cut`` contract for the configured cut.

Status identifiers do not themselves enable CPU fallback. The native adapter
currently maps non-continuous ``EnergyOutOfRange`` results to
``RateEnergyOutOfRange``, including loss queries. Small positive rates remain
valid rates; there is no general low-rate threshold that redirects these
queries to CPU transport.

Final-state capability and CPU completion
-----------------------------------------

`ProcessCapabilities.inl`_ records final-state support independently of rate
availability. In the checked implementation, electron/positron and muon
direct electron-pair production are CPU-only final states: the native
energy-asymmetry :math:`\rho` CDF is unavailable. These channels remain in
reaction selection and are completed by CPU PROPOSAL at the selected vertex;
they are not removed from the physical process set.

Photon conversion to an electron pair, photon conversion to a muon pair and
muon bremsstrahlung have declared backend final-state paths. Their normal
paths use the supported native data and numerical final-state routines.
They can still report invalid input, lookup or kinematic failures. Declared
support is not a guarantee that every request succeeds.

``ProposalCpuFallbackHandler::canHandle`` accepts specific completion
requests with the required interaction identity and vertex state. When required,
the handler first completes the selected loss or replays native selection
using the retained uniform. Replay verifies the interaction hash, process and
component. Every accepted request then calls ``doSpecifiedInteraction`` and
runs the native secondary-processing chain. The relevant contracts are in
`ProposalCpuFallbackHandler.inl`_ and `ProposalFallbackAdapter.inl`_.

An arbitrary lookup failure does not automatically qualify for this handler.
The default hybrid configuration enables ``failOnUnexpectedFallback``;
unexpected requests rejected by the handler can terminate the event rather
than silently repeat it on the CPU. Explicitly disabling that strict policy
allows additional scalar returns. See :doc:`../execution/batches` for the
complete routing policy.

Adding or changing a model
--------------------------

Changes should preserve a documented physical model and make any new data
contract explicit:

1. Configure the native calculator and record its model, component, cut and
   dependency versions. Confirm that the read-only export API exposes all
   required interpolation state.
2. Extend the exporter classifier, record format and validation when the
   model needs new metadata or kinematics. Unsupported tuples must not be
   relabelled as an existing supported model.
3. Update host/device views, hashes, upload and memory accounting together.
   Preserve accumulation order and the distinction between distance and
   channel-selection rates.
4. Define the complete final-state path or an explicit CPU completion policy.
   A rate or loss CDF alone does not establish final-state support.
5. Recreate the backend session after changing its physics configuration;
   event reuse is not an automatic model-invalidation mechanism. Follow the
   cache and lifetime rules in :doc:`preparation` and :doc:`storage`.
6. Check real native tables, boundaries and failure statuses on host and
   device. For a changed physical model, establish independent reference
   expectations for rates, loss sampling and secondary sharing before
   comparing full-shower distributions and process coverage.
   Retain tests for CPU-only channels as well as accelerated channels.

Existing tests are mapped in :ref:`physics-data-validation`. Performance
changes should be measured separately from these correctness contracts.

.. _DiscretePhysicsQueries.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/DiscretePhysicsQueries.hpp
.. _ContinuousPhysicsQueries.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/ContinuousPhysicsQueries.hpp
.. _DiscretePhysicsQueries.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/tables/DiscretePhysicsQueries.inl
.. _TableQueryResult.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/tables/TableQueryResult.hpp
.. _ProcessCapabilities.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/ProcessCapabilities.inl
.. _ProposalCpuFallbackHandler.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/proposal/ProposalCpuFallbackHandler.inl
.. _ProposalFallbackAdapter.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/proposal/ProposalFallbackAdapter.inl
