Photon transport and interaction completion
===========================================

A selected process is not yet an executed interaction. A photon must first
reach its sampled vertex before its final state can be constructed. Charged
leptons additionally update their energy and direction before vertex
selection; that sequence is described in :doc:`lepton-step`.

This page follows the ``proposal-native`` path at the source baseline recorded
in :doc:`../transport`. It distinguishes numerical propagation, final-state
classification, secondary construction and native CPU completion.

Photon propagation
------------------

``transport_photon(environment, config, interaction, external)`` reads the
current photon state and an ``InteractionRecord``. It returns a
``PhotonTransportOutcome`` containing a step record, a fallback request and
``fallbackFlag_``. Inputs are read-only. A partial record can exist on failure;
the caller must check the flag before using it as a completed step.

The interface is declared in `PhotonTransportStep.hpp`_ and implemented in
`PhotonTransportStep.inl`_. Positions and distances are in metres, times in
seconds, total particle energies in GeV and grammage in g/cm².

1. Check the particle identity and selection status, then resolve its current
   environment node. An existing ``ParticleCut`` selection produces a
   zero-length step and records the photon energy as deposited energy.
2. Find the next straight-line environment boundary and observation-plane
   intersection. An optional ``ExternalTransportBoundary`` can impose a
   closer material interface or disable the observation plane. The standard
   resident photon front calls the function with its default external boundary.
3. Integrate the grammage along the ray to the limiting boundary. Compare it
   with the sampled interaction grammage. If the interaction wins, invert
   grammage to distance using ``environment_arclength``.
4. Advance position by :math:`L\boldsymbol{n}` and time by :math:`L/c`.
   Resolve the vertex density for an interaction, or update the node and medium
   when an environment boundary is crossed. Apply the time cut before marking
   an interaction vertex as reached.

The photon direction and energy are unchanged by this propagation function.
It draws no random numbers, constructs no secondaries and invokes no CPU
handler. The selected depth, process and random identities come from the
preceding selection stage.

The comparisons deliberately define exact ties: the environment boundary
wins a tie with the observation plane; an external interface must be strictly
closer; an interaction wins equality with the limiting boundary grammage.
These rules are covered by `testPhotonTransport.cpp`_. The ``LayerBoundary``
name also covers transitions in the general environment tree, not only
spherical atmospheric layers.

``PhotonFront.inl`` calls selection and propagation in separate kernels.
It then builds a stable list of sources whose limit is ``Interaction``.
Only this list enters the final-state classifier. Boundary continuations,
observations, cuts and propagation failures therefore do not execute the
heavy final-state calculation. See `PhotonFront.inl`_ for the actual dispatch
and :doc:`../execution/batches` for returned-record handling.

Final-state interfaces
----------------------

The common pattern is classification followed by writing results to assigned
output slots. The functions are declared in `PhotonFinalStateStep.hpp`_ and
`LeptonFinalStateStep.hpp`_.

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Interface
     - Input
     - Result
   * - ``classify_photon_final_state``
     - Native table view, photon-pair auxiliary data, thinning configuration,
       reached-vertex record, seed and shower ID.
     - Sampled parameters, retained-child count and flags for a final state,
       continuation, suppression or fallback.
   * - ``classify_lepton_final_state``
     - Bremsstrahlung auxiliary data, optional prepared LPM factors, thinning
       configuration, vertex record, seed and shower ID.
     - The corresponding charged-lepton classification. The vertex record
       already contains the selected loss fraction.
   * - ``materialize_photon_final_state_in_place`` /
       ``materialize_lepton_final_state_in_place``
     - Vertex, classification, child offset, first available history ID and
       an output writer; the lepton interface also takes the electron mass.
     - Secondary states and reaction records written to exclusive slots,
       or the specified continuation, suppression or fallback record.

Classification samples the remaining random variables and determines how many
children will be retained. A stable scan assigns output offsets. Materialization
uses those offsets to assign history IDs and write the result; it does not
allocate a queue or call a CPU model. Aggregate-return versions are also
available for standalone callers and tests. Resident execution uses the
in-place interfaces to avoid a large thread-local result object.

New children inherit the vertex position, time and environment association.
Their parent history is recorded, generation is incremented and step ID is
reset. Final-state records preserve the random draw identities and thinning
information. For a generation-zero parent, the first-interaction record
retains the children with the original parent weight before thinning.

Implemented reaction paths
--------------------------

The dispatch table in `ProcessCapabilities.inl`_ is separate from rate-table
availability. The table below describes the final-state implementation;
it does not enable a channel absent from the configured physical model.

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Reaction
     - Completion path
     - Implementation
   * - Electron/positron and muon bremsstrahlung
     - Backend
     - Selected energy loss, local angular calculation, species-specific
       LPM data and suppression decision.
   * - Discrete ionization by a charged lepton
     - Backend
     - Outgoing lepton and delta electron from the selected loss, with
       process-specific kinematics.
   * - Positron annihilation
     - Backend
     - Two-photon energy sharing and directions. The optional stopped-positron
       path uses an explicit ``AtRestAnnihilation`` status.
   * - Photon conversion to an electron pair
     - Backend
     - Native CDF inversion for energy sharing, Sauter directions, electron-pair
       LPM decision and applicable EM thinning.
   * - Photon conversion to a muon pair
     - Backend
     - Native CDF inversion for :math:`E_{\mu^+}/E_\gamma` and BKK directions.
       No electron-pair LPM veto or EM thinning is applied.
   * - Compton scattering
     - Backend
     - Selected photon energy loss and local photon/electron kinematics,
       followed by applicable EM thinning.
   * - Photoelectric absorption
     - Backend
     - Binding-energy correction and one electron along the incoming photon
       direction, using the current native-model convention.
   * - Direct electron-pair production by :math:`e^\pm` or :math:`\mu^\pm`
     - CPU PROPOSAL
     - Complete the selected vertex on CPU because the native energy-asymmetry
       :math:`\rho` CDF is unavailable; the reaction remains in selection.

The registry also declares CPU-only identifiers for photonuclear reactions,
lepton muon-pair production, hadronic production, weak interactions,
photoproduction and decay. Their availability depends on the configured
models. Transport-selected muon decay has a separate forced-decay return
path, described in :doc:`lepton-step`; it is not ordinary final-state
materialization in these kernels. An unregistered process is reported as
``GpuProcessNotImplemented`` and must not silently become an accepted CPU
reaction.

The current pair-production paths do not use rejection sampling to replace
native CDF inversion or the missing :math:`\rho` distribution. Some older
pair-related identifiers and materialization branches remain in the source;
actual support is determined by the capability dispatch and classifier.
See :doc:`../physics-data/contracts` for the distinction between a loss CDF
and a complete final-state model.

LPM suppression and thinning
----------------------------

For bremsstrahlung and photon electron-pair production, a valid sampled
candidate can be rejected by the physical LPM survival decision. This is
distinct from a failed numerical query. The result keeps the parent at the
reached vertex, increments its step ID and returns a suppression record;
no children are constructed. Propagation already completed before that
vertex is retained.

Thinning is applied after the applicable LPM decision. The retained-child
count, weights and keep mask must remain consistent with the scan and output
writer. Muon bremsstrahlung bypasses EM thinning through the charged-lepton
helper, and photon muon-pair production retains both children with the
parent weight. The complete order is defined in `LeptonFinalStateStep.inl`_
and `PhotonFinalStateStep.inl`_.

Native mass conventions and the muon-pair cut
---------------------------------------------

PROPOSAL and C8 use slightly different numerical mass constants. The helper
``proposal_secondary_transport_energy_gev`` preserves the sampled kinetic
energy when converting a secondary's total energy to the transport mass
convention. Reaction records carry the corresponding weighted mass-convention
correction for energy accounting; see `TransportMass.inl`_. This conversion
must remain consistent with secondary cuts and output accumulation.

The native photon-to-muon-pair model can produce a positive sampled secondary
energy at or below the muon rest mass. The materializer marks such a converted
child with ``NativeMuonPairCutFlag``. This is a narrowly defined compatibility
case: a newly created muon, generation greater than zero, step ID zero and
positive total energy no greater than the transport mass.

The following lepton step recognizes the tag and produces a zero-length cut
record. It preserves the native signed kinetic-energy deposit
:math:`E_\mu-m_\mu c^2`, including a negative value for this specific case.
It neither propagates that child as an ordinary muon nor redirects the whole
photon interaction to CPU. `NativeMuonPairCut.inl`_ and
`NativeMuonPairCutRecord.inl`_ implement this contract; the tag does not
permit arbitrary sub-rest-mass input states.

Specified CPU completion
------------------------

CPU-only final states retain the reached vertex and selected process/component
identity. Photon selection can store a deferred completion reason in its
interaction record; photon propagation must reach the vertex before that
request becomes a specified CPU interaction. A return before reaching the
vertex is not interchangeable with this request.

``ProposalCpuFallbackHandler::canHandle`` checks whether the request contains
an accepted specified interaction or a resolvable selected loss.
``handle(stack, event)`` then:

1. Imports the retained parent into the native stack and resolves its node.
   The CPU calculator key comes from that node's nuclear composition, not
   from the PROPOSAL medium hash stored in table metadata.
2. Reconstructs the interaction record. If required, it completes the selected
   loss or replays native selection with the retained uniform, checking the
   selected identities.
3. Supplies the required final-state random numbers to
   ``doSpecifiedInteraction`` and verifies successful completion.
4. Calls ``processList_.doSecondaries(secondaries)`` for native secondary
   processing, then erases the parent and updates completion statistics.

The implementation is in `ProposalCpuFallbackHandler.inl`_. Completion avoids
resampling the physical interaction distance, but the native secondary chain
can still consume its own random stream. It must not be treated as a
random-number-free operation.

Creating a ``FallbackRequest`` alone does not execute this handler or authorize
arbitrary numerical failures as CPU work. The standard router's strict policy
can reject unexpected requests. Scheduling and acceptance rules are described
in :doc:`../execution/batches`; numerical statuses are described in
:doc:`../physics-data/contracts`.

Validation coverage
-------------------

`testPhotonTransport.cpp`_ covers exact boundary ties, zero-length interactions,
node/material transitions, time cuts, failed inverse grammage and invalid
vertex density. It compares current host and selected-execution-space paths
with a retained production reference and checks that the input is unchanged.

`testPhotonFinalState.cpp`_ and `testLeptonFinalState.cpp`_ check routing flags,
invalid inputs, child identities, thinning, LPM suppression and final-state
records. Real-table cases are enabled when ``CORSIKA_ENABLE_PROPOSAL_EXPORT``
is set. The photon tests separately exercise the muon-pair cut tag and energy
convention. `testTransportRecords.cpp`_ checks the corresponding cut records.

`testCpuFallbackPhysics.cpp`_ exercises native specified reactions and selection
replay, including direct electron-pair production by both electron signs and
both muon signs. Comparisons with a retained production implementation are
conditional on the reference build option. These checks establish regression
contracts within their test configurations.
Independent physical validation and full-shower statistical comparisons remain
separate requirements for changes to the model or execution path.

.. _PhotonTransportStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/PhotonTransportStep.hpp
.. _PhotonTransportStep.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/PhotonTransportStep.inl
.. _PhotonFront.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/PhotonFront.inl
.. _PhotonFinalStateStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/PhotonFinalStateStep.hpp
.. _LeptonFinalStateStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/LeptonFinalStateStep.hpp
.. _ProcessCapabilities.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/ProcessCapabilities.inl
.. _LeptonFinalStateStep.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/LeptonFinalStateStep.inl
.. _PhotonFinalStateStep.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/PhotonFinalStateStep.inl
.. _TransportMass.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/TransportMass.inl
.. _NativeMuonPairCut.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/NativeMuonPairCut.inl
.. _NativeMuonPairCutRecord.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/NativeMuonPairCutRecord.inl
.. _ProposalCpuFallbackHandler.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/proposal/ProposalCpuFallbackHandler.inl
.. _testPhotonTransport.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/transport/testPhotonTransport.cpp
.. _testPhotonFinalState.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/finalstate/testPhotonFinalState.cpp
.. _testLeptonFinalState.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/finalstate/testLeptonFinalState.cpp
.. _testTransportRecords.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/transport/testTransportRecords.cpp
.. _testCpuFallbackPhysics.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/proposal/testCpuFallbackPhysics.cpp
