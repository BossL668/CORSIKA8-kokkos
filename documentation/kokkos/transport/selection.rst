Candidate sampling and vertex selection
=======================================

An interaction depth is a candidate for the next step, not a complete
transport decision. Geometry, continuous loss, magnetic deflection and muon
decay can stop a particle earlier. For charged leptons, the energy at the
candidate vertex also differs from the energy used to sample that depth.

The two selection interfaces below preserve this distinction. They operate
on value records and borrowed physics views. They neither move particles nor
allocate or compact execution queues.

Selection interfaces
--------------------

.. list-table::
   :header-rows: 1
   :widths: 29 35 36

   * - Function
     - Input
     - Result
   * - ``select_discrete_interaction``
     - ``InteractionPhysicsView``, initial ``ParticleState``, input index,
       random seed and shower ID.
     - ``InteractionSelectionOutcome``: candidate interaction record or
       fallback request, with native inverse counters.
   * - ``select_lepton_vertex``
     - The same physics view and event key; an ``InteractionRecord`` at the
       transported, post-loss vertex.
     - ``LeptonVertexSelectionOutcome``: exactly one interaction,
       continuation or fallback branch, with native inverse counters.

Particle records use total energy in GeV. Selection converts it to MeV for
the native tables. Rates have units cm²/g and interaction depth has units
g/cm². The input index identifies the caller's result slot; it is not a
random-number input. See `InteractionSelection.hpp`_,
`LeptonVertexSelection.hpp`_ and :doc:`../particles/representation`.

Initial rate and candidate depth
--------------------------------

``select_discrete_interaction`` first handles the native tagged muon-pair
cut, the particle-time cut, and species-specific energy cuts. An ordinary
charged-lepton cut uses kinetic energy relative to the transport mass and
also checks the continuous table's lower endpoint. The mass used by a native
table and the mass used for transport are separate query results; they must
not be replaced by an assumed common value.

For a valid muon above its transport rest mass, the function also samples a
laboratory decay distance and retains it in ``decayDistanceM_``. Transport
later compares that distance with the other limits. Selection does not
execute the decay.

For a positive total rate, the candidate depth is

.. math::

   \Delta X_{\mathrm{int}} = -\frac{\ln U_d}{r_{\mathrm{tot}}(E_i)},
   \qquad 0 < U_d < 1.

Here :math:`r_{\mathrm{tot}}` is the independently exported total-rate
interpolant. A successful nonpositive rate produces
``NoDiscreteInteraction`` with infinite candidate depth. A failed rate query
instead returns a fallback request with the table status; these are different
outcomes.

The exact draw and record assignment are:

.. literalinclude:: ../../../corsika/detail/kokkos/em/algorithms/InteractionSelection.inl
   :language: cpp
   :lines: 104-111
   :lineno-match:

For charged leptons, this function then returns ``DistanceSampled``. No
process, target component or stochastic loss fraction has yet been selected.
For photons, energy is unchanged by flight, so channel and loss selection can
already be prepared at this stage. A prepared photon interaction still has
to compete with geometry and reach its vertex; see :doc:`interactions`.

Charged-lepton vertex acceptance
--------------------------------

After a successful step reaches the candidate interaction, the transport
record carries ``RequiresReselection`` and the post-continuous-loss energy
:math:`E_f`. The resident path also completes multiple scattering before
calling ``select_lepton_vertex``. The name ``RequiresReselection`` is a stage
label; it does not mean that the current initial sampler already chose a
charged-lepton reaction.

The vertex selector retains the earlier acceptance uniform :math:`U_a` and
initial rate, then evaluates the total rate at :math:`E_f`. It accepts the
candidate only when

.. math::

   U_a\,r_{\mathrm{tot}}(E_i) < r_{\mathrm{tot}}(E_f).

The comparison is strict. A failed acceptance produces
``NoDiscreteInteraction`` at the advanced particle state, clears the reaction
fields and increments ``stepId_`` for continuation. It does not undo the
completed transport or continuous energy loss. An exhausted step-ID range
returns a failure instead of wrapping around.

An accepted candidate then receives an independent native-selection uniform.
This ordering is implemented in `LeptonVertexSelection.inl`_; a native
at-rest positron-annihilation record has a separate early branch.

Ordered channel and loss selection
----------------------------------

``select_rate_column_by_uniform`` selects a process and target-component
column in the exported native order. Its normalization is the ordered sum of
individual column rates, which is distinct from the total-rate interpolant
used for distance and outer acceptance. This distinction is documented in
:doc:`../physics-data/storage`.

The implementation sets ``sampled_rate`` to the selection uniform times that
column sum, subtracts each column rate in order, and selects the first column
that makes the residual negative. It returns
``lossQuantile_ = -sampled_rate / selected_rate`` for that interval. The
subtraction order, strict comparison and residual orientation are part of the
native selection contract.

The selected quantile is then passed to the native loss-fraction query.
Success supplies ``processId_``, ``componentHash_`` and ``energyFraction_``
in a ``Selected`` record. It does not by itself construct directions or
secondary particles. Selected CPU-only processes and unavailable inverses
follow the explicit policies in :doc:`../physics-data/contracts` and
:doc:`interactions`.

This vertex acceptance step is separate from rejection sampling of a
final-state energy split. The standard native-table path described here uses
CDF inversion where the required distribution is available.

Random provenance and stage records
-----------------------------------

``RandomNumberKey`` combines the event seed, shower ID, particle history ID,
step ID, process domain and draw ID. ``uniform_open01`` evaluates the
counter-based Philox generator from that identity. Queue position and thread
index do not enter the key. Moving a particle to a different execution slot
therefore does not change this draw when the full identity remains unchanged.
This does not imply that changing event scheduling or CPU random-stream
history preserves a whole shower bit for bit.

The separate domains in `RandomDomains.hpp`_ distinguish interaction distance,
outer acceptance, native channel selection, scattering and muon decay.
``InteractionRecord`` retains uniforms, domain identifiers, target identity
and the vertex-reached flag for downstream processing and CPU completion.
Consumers must use the outcome flags and stage fields together; the presence
of a process ID alone is not proof that an interaction has occurred.

Inverse iteration counters describe the table query performed at this stage.
They do not count accepted reactions or transported steps. Preserve these
distinctions when adding diagnostics or changing kernel boundaries.

Validation coverage
-------------------

`testSelectionContracts.cpp`_ exercises early cuts, invalid input, acceptance
ties, rejected-vertex continuation, step overflow, missing inverses and
deferred photon completion with controlled tables. The controlled tables
exercise branches; they are not physical cross-section models.

`testInteractionSelection.cpp`_ uses real native tables and compares host and
selected-execution-space results against the retained production reference.
It covers table endpoints, accepted and rejected charged-lepton vertices,
muon decay draws and direct-pair replay. Its prescribed post-transport input
does not independently validate geometry or continuous propagation; those
are covered in :doc:`lepton-step`.

The real-table target is conditional on ``CORSIKA_ENABLE_PROPOSAL_EXPORT``.
These source references describe test coverage, not a claim that the tests
have run on every device.

.. _InteractionSelection.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/InteractionSelection.hpp
.. _LeptonVertexSelection.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/LeptonVertexSelection.hpp
.. _LeptonVertexSelection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/algorithms/LeptonVertexSelection.inl
.. _RandomDomains.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/random/RandomDomains.hpp
.. _testSelectionContracts.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/selection/testSelectionContracts.cpp
.. _testInteractionSelection.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/selection/testInteractionSelection.cpp
