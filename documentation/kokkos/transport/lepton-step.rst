Charged-lepton propagation
==========================

``transport_lepton(...)`` advances an electron, positron or charged muon from
an ``InteractionRecord`` to a ``LeptonTransportRecord``. It determines the
limiting distance, advances the particle in the local magnetic field and
computes continuous energy loss. Multiple scattering is a separate stage;
interaction-channel selection at the endpoint and final-state generation
follow it. The interfaces are declared in `LeptonTransportStep.hpp`_ and
`MoliereStep.hpp`_.

The resident path launches selection, transport and scattering on the same
execution-space instance in that order. A common source index connects their
records. ``transport_lepton`` therefore receives a sampled candidate depth,
not a final step length or an already completed propagation. See
:doc:`selection` for the preceding sampling and subsequent vertex selection,
and :doc:`../execution` for batch scheduling.

Inputs, results and execution status
------------------------------------

The table and environment arguments are borrowed views. Their owners must
remain alive in memory accessible to the executing kernel. Positions,
directions, magnetic-field components and the observation plane must use the
same coordinate system. No native C8 environment object is constructed inside
the particle step.

.. list-table::
   :header-rows: 1
   :widths: 29 71

   * - Argument or result
     - Contract
   * - ``ContinuousPhysicsView table``
     - Native range and loss queries, particle masses and kinetic transport
       cuts. Table energies are MeV; particle-state energies are total energy
       in GeV.
   * - ``environment`` and ``TransportGeometryConfig config``
     - Environment-tree queries, density and magnetic field, geometric
       limits and a separate observation plane. Positions and lengths use
       metres; grammage uses g/cm².
   * - ``InteractionRecord interaction``
     - The starting particle, sampled interaction grammage and, for muons,
       a possible decay distance. Accepted input states are ``Selected``,
       ``DistanceSampled``, ``NoDiscreteInteraction`` and ``ParticleCut``.
   * - ``ExternalTransportBoundary external``
     - An optional non-absorbing material-interface distance expressed in
       the same leapfrog step-length parameter as transport. Its flag can
       disable the separate observation-plane test.
   * - ``LeptonTransportRecord record``
     - Start and endpoint states, the limiting process, traversed grammage,
       separate continuous and cut deposits, and magnetic/scattering
       diagnostics. The record definition is in `LeptonTransportRecord.hpp`_.
   * - ``FallbackRequest fallback`` and return value
     - Return ``1`` supplies a fallback request; the output record may still
       contain partial writes. Return ``0`` indicates successful transport.
       Return ``2`` is a successful zero-length cut that the split scattering
       stage must bypass and normalize to ``0``.

The ``applyMoliere`` argument selects the zero-length-cut return convention;
it does not make ``transport_lepton`` execute scattering. Inputs and output
slots must not alias. Consumers must inspect the status before treating the
record as a completed step. Returning a fallback request also does not invoke
the CPU handler or override its acceptance policy.

Step-limit competition
----------------------

`LeptonTransportStep.inl`_ performs the following operations before advancing
the particle:

1. ``prepare_lepton_continuous_step`` resolves the mass, transport cut and
   native table minimum. For a propagating particle it queries the range at
   the initial total energy and at the larger of ``0.9 * initial_energy_MeV``
   and the table minimum. Their difference is the maximum continuous-loss
   grammage. The 10% limit applies to **total energy**, while the particle cut
   applies to **kinetic energy**. `LeptonContinuousStep.inl`_ retains the
   native query order and the existing roundoff correction for a tiny
   negative range difference.
2. ``query_environment`` resolves the current node. Its model supplies the
   local magnetic field. ``maximum_uniform_magnetic_step`` computes the
   magnetic-deflection limit; ``lepton_boundary`` and
   ``lepton_observation_distance`` find the competing tree-boundary and
   observation-plane distances. Observation is not a material-tree node.
3. The nearest applicable geometric candidate is converted to grammage
   along the **starting direction** by ``geometry_competition_grammage``.
   This candidate competes with the continuous-loss grammage and sampled
   interaction grammage. If loss or interaction wins,
   ``environment_arclength`` converts that grammage to a length using the
   same starting position and direction.
4. A finite, non-negative sampled muon decay distance competes with the
   selected length. If it is shorter or equal, the decay candidate determines
   the endpoint.

The comparisons are part of the implementation contract. Continuous loss
wins only when its grammage is strictly smaller than both other candidates.
Otherwise an available interaction wins if its grammage is no larger than
the geometric grammage. Muon decay wins equality with the resulting length,
including a zero-length candidate. Replacing these comparisons with an
arbitrary ordering of a minimum operation can change endpoint decisions.

The boundary calculation uses the magnetic trajectory policy before the
final distance is known. It is not a straight-ray boundary search followed
by unrelated bending. The straight-direction assumption above concerns the
grammage-to-length conversion for limit competition. The shared node and
boundary interfaces are described in :doc:`../environment/queries`.

Magnetic advance and continuous loss
------------------------------------

``advance_uniform_magnetic_field`` updates the position, tracked direction
and time for the chosen length. When the preceding magnetic policy selected
``MagneticStepStatus::Linear``, transport passes a zero field to this update.
That policy also covers sufficiently large gyroradius or sufficiently small
transverse momentum, not only an exactly zero physical field. It prevents
the propagation update from reintroducing bending after a straight boundary
intersection. The environment's stored field is not modified.

After the endpoint is known, continuous loss uses the **straight chord
between the start and endpoint**, with its own direction and length. It
does not reuse the starting-direction grammage used in step competition,
nor integrate the two magnetic half-steps separately. This follows the
scalar continuous-process use of the completed step's straight track.

The chord query result is passed to range inversion and recorded explicitly:

.. literalinclude:: ../../../corsika/detail/kokkos/em/algorithms/LeptonTransportStep.inl
   :language: cpp
   :lines: 332-342
   :lineno-match:

``evaluate_lepton_continuous_loss`` returns the post-step total energy in
MeV and the unweighted continuous deposit in GeV. A
``TransportCutReached`` query result sets the endpoint energy to the table
minimum and marks the cut. Other unsuccessful inverse results become
fallback requests with query-stage diagnostics. If the continuous-loss
limit won, a non-positive actual energy loss is also an explicit failure.

Multiple scattering
-------------------

``apply_moliere_scattering_stage`` consumes the completed transport record.
It selects the electron or muon ``MoliereSnapshot`` from the starting PID,
so muons use their own mass and material parameters. A missing required
muon snapshot produces ``MoliereParametersUnavailable``. An incoming
transport failure passes through; state ``2`` bypasses sampling and becomes
successful state ``0``.

The stage generates two projected-angle variates and one azimuth variate
from the starting particle's history and step IDs, the shower ID, seed and
dedicated draw IDs. These draws do not depend on the particle's current
queue position. The sampler uses the chord grammage and initial energy;
although its interface also accepts the final energy, the current
implementation explicitly leaves that argument unused. It samples two
one-dimensional projected angles and combines their squares to obtain the
scattering-angle magnitude. See `MoliereStep.inl`_ and
`MoliereScattering.inl`_.

``apply_moliere_direction_cpu_step_compatible`` first constructs a scattered
direction relative to the starting direction. It adds that direction change
to the magnetically tracked direction and normalizes the result, matching
the scalar step-update convention. Scattering changes the endpoint
**direction**, not the already computed endpoint position, time or loss.
For an interaction or decay candidate, the updated endpoint is then copied
into ``record.interaction_.particle_`` before downstream processing.

The resident implementation chooses a four-component or general-capacity
kernel for the same scattering algorithm. These are storage-capacity
specializations, not separate electron and muon transport queues.

Endpoint outcomes and observable records
----------------------------------------

The record's ``limit_`` determines subsequent processing in
`LeptonFront.inl`_:

* ``ContinuousStep``, ``MagneticStep``, ``LayerBoundary`` and
  ``MaterialBoundary`` retain a particle for further transport. A tree
  crossing updates the node and medium identity; an external material
  interface remains distinct from an absorbing observation surface.
* ``InteractionCandidate`` marks a reached vertex and changes the nested
  interaction state to ``RequiresReselection``. Channel and loss selection
  must use the endpoint after continuous loss and scattering.
* ``DecayCandidate`` supplies the post-step muon to the decay output. The
  host router in `EmRouter.inl`_ returns it to the native stack as a forced decay; device
  transport does not construct the decay products. This is separate from a
  failed table query or unsupported electromagnetic final state.
* ``ObservationSurface`` and ``EscapedEnvironment`` produce terminal
  observation records with different statuses. ``ParticleCut`` terminates
  the particle and retains the cut deposit.

Ordinary continuations and cuts advance ``stepId_`` in transport. Interaction
and decay candidates retain the starting step ID while their endpoint
decision is resolved. After successful transport and scattering, the completed
step remains an output even if later vertex selection or final-state generation
requires CPU completion; it must not be propagated or deposited a second time.

A particle already below its transport cut terminates in place. The
special tagged native muon-pair cut also produces a zero-length record,
retaining the scalar cut's signed kinetic-energy accounting. A particle
that crosses an energy or 10 ms time cut during an otherwise finite step
terminates at the **completed endpoint**: transport does not truncate the
step at exactly the cut time. The pending interaction or decay is suppressed.
If that step also reaches the observation surface,
``observationSurfaceReachedBeforeCut_`` preserves both outcomes. Stopped
positron annihilation is disabled by the fixed production policy at this
chapter's baseline; stochastic in-flight annihilation is a separate process.

Continuous and cut deposits remain separate and unweighted in the record.
Profile projection combines them while retaining the statistical weight
separately. The radio projection constructs finite electron/positron tracks
from start/end positions and times; it does not substitute the scattered
endpoint direction for the completed track displacement. Zero-length cuts
therefore do not generate a finite radio track. See `ProfileProjection.inl`_
and `RadioProjection.inl`_.

Regression coverage
-------------------

`testLeptonTransport.cpp`_ compares the current single-step routine with a
frozen production reference on the host and in the selected execution space.
It checks complete result/fallback records, unchanged inputs, zero-length
cuts and the absence of scattering inside ``transport_lepton``. With
PROPOSAL export enabled, it additionally covers the real-table energy/depth
grid, curved and linear policies, external interfaces, all transport limits,
observation-plus-cut outcomes and exact interaction/decay ties.

`testMoliereStep.cpp`_ checks production draw identities, projected-angle
sampling, direction updates, bypass states, missing muon parameters and
failure provenance. Its real-table cases are also conditional on PROPOSAL
export. `testContinuousContracts.cpp`_ covers cut comparisons, continuous
preparation/loss results and fallback provenance.

These tests establish regression coverage against the recorded implementation.
They do not by themselves establish independent physical accuracy, equality
between different devices, or a whole-shower statistical comparison. The
build conditions for the transport tests are in `transport/CMakeLists.txt`_.

.. _LeptonTransportStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/LeptonTransportStep.hpp
.. _LeptonTransportStep.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/LeptonTransportStep.inl
.. _LeptonContinuousStep.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/LeptonContinuousStep.inl
.. _LeptonTransportRecord.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/LeptonTransportRecord.hpp
.. _MoliereStep.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/MoliereStep.hpp
.. _MoliereStep.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/MoliereStep.inl
.. _MoliereScattering.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/models/MoliereScattering.inl
.. _LeptonFront.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/LeptonFront.inl
.. _EmRouter.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/EmRouter.inl
.. _ProfileProjection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/output/algorithms/ProfileProjection.inl
.. _RadioProjection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/algorithms/RadioProjection.inl
.. _testLeptonTransport.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/transport/testLeptonTransport.cpp
.. _testMoliereStep.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/transport/testMoliereStep.cpp
.. _testContinuousContracts.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/continuous/testContinuousContracts.cpp
.. _transport/CMakeLists.txt: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/transport/CMakeLists.txt
