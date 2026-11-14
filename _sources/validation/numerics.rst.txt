Numerical and statistical validation
====================================

Validation asks three different questions: whether a change preserves a
specified calculation, whether the calculation satisfies its physical and
accounting contracts, and whether complete showers have compatible
distributions. Each requires its own reference and acceptance rule. Agreement
with a frozen implementation establishes a regression result; it does not
independently establish the accuracy of the underlying model.

The implemented checks below refer to source revision
``d5297c43c041a22b43c65c322d9b51bc7ec8e397``. They describe test assertions,
not results from a new test run. The complete-shower protocol at the end is a
recommended procedure, separate from those assertions.

Align the physical problem
--------------------------

Before comparing backends, retain the resolved configuration and match the
following quantities. :doc:`../usage/configuration` describes the application
options; `c8_air_shower.cpp`_ supplies the air application's construction and
process order.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Configuration
     - Required agreement
   * - Primary
     - Particle or nuclear species, total energy or energy-distribution law,
       injection position, direction, time and forced-interaction/decay policy.
       The air application's primary energy is total energy in GeV; for a
       nucleus it is the whole nucleus's energy.
   * - Transport and cuts
     - Kinetic tracking cuts for EM particles, hadrons, muons and taus;
       neutrino tracking; time cuts; magnetic deflection and continuous-step
       policies. The air application sets the PROPOSAL production threshold
       to the minimum of its four tracking-cut options. Equal primary energy
       does not compensate for unequal production thresholds.
   * - Stopped positrons
     - The native/PROPOSAL transport policy has
       ``EnableStoppedPositronAnnihilation = false`` in `PositronAtRest.hpp`_.
       This controls extra photons from positrons removed at the cut;
       in-flight annihilation remains a separate process. Do not assume an
       independently configured EM model has the same termination policy.
   * - Models and tables
     - EM model, hadronic and decay models, high-/low-energy model transition,
       model-data versions, material composition and density, production
       thresholds, native/auxiliary table identities and interpolation
       settings. Changing from PROPOSAL to EGS4 is a model comparison as well
       as an execution comparison. See :doc:`../physics-data`.
   * - Thinning
     - Threshold, maximum weight and zero-weight-particle policy. In the air
       application, ``--emthin 0`` disables thinning; ``--max-weight 0`` selects
       an automatic weight limit. The default thinning fraction is nonzero.
       Weighted particle sums and their fluctuations must be interpreted
       using the actual thinning configuration.
   * - Environment and observation
     - Atmosphere or material profile, interfaces, magnetic field, coordinate
       system, injection geometry, observation surface and escape boundaries.
       A material interface and an absorbing observation plane have different
       meanings. Match longitudinal depth bins and ``--profile-crossings``
       (``both``, ``forward`` or ``original-c8``).
   * - Radio
     - Emission formalism, propagation model and table fingerprint, observer
       positions, component basis, time origin, sample rate, duration,
       frequency grid where applicable, refinement and accumulation policy.
       Match reflection/branch choices for firn. An empty observer list or a
       truncated time window cannot validate a radio signal.

Keep execution choices alongside this physics record: backend, compiler,
floating-point options, device architecture, thread count, batching, CPU
fallback policy, worker partitioning and lifetime. Some choices can change
rounding, scheduling or random-stream progression without changing the
intended physics.

Implemented unit and module checks
----------------------------------

The suite uses several kinds of reference. Read the assertion and fixture
together before interpreting a passing target.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Check
     - Scope of the assertion
   * - `testStochasticContracts.cpp`_
     - Philox golden words, production process/draw identities, every bit of
       the key fields, reversed device traversal, host/device random results
       and open-interval uniforms. This tests the implemented random mapping
       and routing contracts, not shower-distribution agreement.
   * - `ContinuousChecks.hpp`_ and continuous query tests
     - Current and frozen production queries run on the same execution space.
       Floating results use exact equality, with signed-zero checks and
       matching NaN classification; state fields also match exactly.
       Native-model diagnostic errors are reported separately and explicitly
       identified as **not** migration tolerances. Real-table query targets
       require ``CORSIKA_ENABLE_PROPOSAL_EXPORT``; see
       `continuous/CMakeLists.txt`_.
   * - `testPhotonCascade.cpp`_ and `testLeptonCascade.cpp`_
     - Queue ownership, pending inputs, checkpoints, histories and multi-front
       execution against reference drivers. Native-table cases are
       conditional on PROPOSAL export. These are component cascades and
       scheduling regressions, not an ensemble of complete application
       showers.
   * - `testProfileAccumulation.cpp`_ and `testLeptonStatistics.cpp`_
     - Profile and ledger accumulation, known totals, fixed-point overflow,
       counters, and fused versus separate accumulation. The material-boundary
       fixture in `testMaterialBoundaryStatistics.cpp`_ uses real firn
       geometry with an exact linear range fixture to isolate record
       accounting from cross-section accuracy.
   * - `testRadioProjection.cpp`_ and `testRadioAccumulator.cpp`_
     - Projection against the frozen implementation and native C8 CoREAS/ZHS
       modules; deterministic and floating accumulation, capacity and
       rejection contracts. `testFullWaveAccumulator.cpp`_ separately compares
       firn full-wave spectra with a CPU response calculation on a supplied
       or generated response-table fixture.

Some concrete tolerances illustrate why one global numerical threshold is
inappropriate:

* The firn material-boundary fixture requires distance error below
  :math:`10^{-12}` m and grammage error below :math:`10^{-10}` g/cm².
  Its fused and separate fixed-point histogram and counter results are equal.
* The photon muon-pair materialization case in `testPhotonFinalState.cpp`_
  checks the weighted child energy **after subtracting the recorded mass
  convention correction** against the weighted parent energy, with a bound
  of :math:`8\epsilon` times that parent energy. Here :math:`\epsilon` is
  ``std::numeric_limits<double>::epsilon()``.
* The native radio projection comparison bounds the maximum sample error in
  each observer/component by ``floor + 2e-8 * reference_peak``. The floor is
  :math:`2\times10^{-18}` for CoREAS electric field and
  :math:`2\times10^{-27}` for ZHS vector potential, in their respective SI
  units. The separate comparison against frozen projection code is exact.
* The floating radio accumulator comparison uses
  ``1e-28 + 2e-12 * reference_peak`` per component. Deterministic waveform
  comparison and exported fixed-point arrays use equality. This statement
  concerns the test's current/reference pair on the same execution space;
  it is not a cross-device bitwise guarantee.
* The firn full-wave fixture requires relative :math:`L_2` error below
  :math:`2\times10^{-10}` for total, direct and boundary complex spectra.
  Its definition is
  :math:`\sqrt{\sum|C-R|^2/\max(\sum|R|^2,10^{-300})}` over observers,
  components and frequencies. It also checks reset/replay and batch-size
  equivalence for that fixture.

These bounds belong to the named tests and quantities. They are neither
universal physics-error estimates nor acceptance thresholds for arbitrary
complete-shower outputs. Fixed-point addition removes summation-order
dependence for identical quantized contributions within range; it cannot
make different trajectories or differently rounded contributions identical.

Random streams and reproducibility
----------------------------------

A seed is only one part of a reproducibility record. The Kokkos EM
``RandomNumberKey`` contains seed, shower ID, particle history ID, step ID,
process ID and draw ID. `Philox.hpp`_ and `Philox.inl`_ map this identity to
random words without using the particle's queue position. Changing a process
decision, particle genealogy or draw identity can still change subsequent
histories.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Execution mode
     - Stream policy to record
   * - Direct application, ``-N > 1``
     - Native CPU streams are seeded once before the event loop and continue
       across events. The direct loop supplies successive shower IDs to
       execution. It does not reseed the native RNG with ``seed + event_index``.
   * - Coordinator events
     - The coordinator uses ``first_seed + event_index`` for successive event
       seeds. ``worker_seed(seed, index)`` hashes the decimal ``seed:index``
       string with SHA-256, takes the first eight bytes little-endian, masks
       to 63 bits and replaces zero by one. GPU worker indices start at zero;
       parallel CPU-prefix worker indices use ``1024 + i``. See
       `AirShowerCoordinator.cpp`_ and `StaticFrontierIO.cpp`_.
   * - Persistent GPU workers
     - Each request sets the seed for device sampling and selected CPU
       fallback replay. Native CPU streams, copied model callbacks and
       SOPHIA's private stream continue from worker initialization. They are
       not rewound to the request seed. `WorkerSession.inl`_ records
       ``continue-worker-streams`` and the CPU initial seed.

Consequently, event two of a direct multi-event process, a fresh process with
an incremented seed, and a request in a persistent worker are different
stream histories. Record the event order, worker IDs, partitions, skips and
reuse policy when reproducing a campaign. `WorkerSeedTest.cpp`_ checks hash
vectors, CLI/JSON preservation and a finite collision-regression sample;
`WorkerSessionTest.cpp`_ checks native and copied-stream continuation. A
finite seed scan does not establish that collisions are impossible.

For deterministic debugging, fix the input particles or track spool, physical
configuration, keys and stream state, then compare decisions, records and
accumulators at the first divergence. `testDecisionReplayTape.cpp`_ tests
decision-tape serialization and preserved wire data. A successful tape
round-trip establishes that diagnostic contract; it is not an assertion that
every backend or scheduler replays a whole shower identically.

Use statistical acceptance when histories or stream policies differ.
Identical top-level seed numbers alone do not justify treating outputs as
event-by-event numerical replicas. See :doc:`../execution/coordinator` for
worker lifecycle and :doc:`../radio/inputs` for radio-track replay.

Energy accounting and numerical residuals
-----------------------------------------

Distinguish a comparison residual from an energy-closure residual. For a
candidate value :math:`C_i` and reference :math:`R_i`, define the signed
comparison residual as :math:`d_i=C_i-R_i`. Report absolute errors in physical
units; for a normalized error, state the denominator and its behavior at
zero. For example,

.. math::

   \begin{aligned}
   e_i &= \frac{|C_i-R_i|}{\max(|R_i|,q_0)},\\[0.6em]
   e_2 &= \frac{\left(\sum_i|C_i-R_i|^2\right)^{1/2}}
                     {\left(\sum_i|R_i|^2\right)^{1/2}}.
   \end{aligned}

The chosen :math:`q_0>0` has the observable's units and must be documented.
The second expression is defined only for a nonzero reference norm; report
an absolute norm for a zero reference. Neither expression supplies an
acceptance threshold by itself. Require finite values before computing these
metrics, unless the particular contract explicitly expects a nonfinite value.

The air application's legacy log reports
:math:`100[(E_{\mathrm{deposit}}+E_{\mathrm{ground}})/E_0-1]`.
This is a signed percentage difference, not the strict EM ledger below.
It omits explicit terms needed by that ledger and cannot be interpreted as
a universal conservation test.

`ShowerReport.inl`_ defines the resident EM ledger directly:

.. literalinclude:: ../../../corsika/detail/kokkos/application/cascade/ShowerReport.inl
   :language: cpp
   :lines: 1242-1257
   :lineno-match:

The signed residual is **source minus terminal**, in GeV. The normalized
closure error takes its absolute value; the denominator's floor is the
smallest positive normal ``double``, not machine epsilon. Medium electron
rest mass is a physical input for the accounted atomic-electron processes.
The signed mass-convention correction records conversion between PROPOSAL
total energies and the C8 kinetic-energy stack. Cut rest mass and otherwise
unwritten photoelectric binding energy are explicit sinks; observation/cut
overlap is subtracted to avoid double counting. `ProfileResult.hpp`_ documents
these terms. A worker's initial energy is the weighted sum of its imported
roots, rather than the global primary energy.

The implemented ``energy_ledger.acceptance_tolerance`` is :math:`10^{-4}`.
``accepted`` additionally requires ``complete_coverage``: an EM primary,
no thinning, no scalar EM steps, no returned CPU fallback or memory spill,
and no specified CPU final states. The report throws when complete coverage
holds and the relative error exceeds this tolerance. If coverage is false,
``accepted: false`` does not by itself diagnose failed energy conservation;
the reported resident ledger is insufficient for that whole shower. Do not
apply this gate to a hadronic or thinned shower without a ledger covering its
remaining processes and weighting conventions.

Recommended complete-shower protocol
------------------------------------

The following procedure is proposed for accepting a physics or backend
change. It is not an implemented universal ensemble gate at this revision.
In particular, the coordinator's `compare.py`_ exercises closed synthetic
inputs and process/output handling, and explicitly makes no shower claim.
``COMPLETE.json`` and successful unit tests are prerequisites for a campaign,
not its statistical conclusion.

1. **Define the claim and reference.** Specify whether the comparison changes
   execution, the physics model or both. Freeze the configuration above and
   choose representative primary species, energies, directions and material
   conditions for the claimed scope. Include boundary and observation cases
   when those paths change. Establish the intended observables, uncertainty
   method, sample size and scientifically meaningful equivalence bounds
   before inspecting candidate results.
2. **Check complete events.** Retain commands, resolved seeds and stream
   policy, software/table hashes, completion records, logs and raw outputs.
   Inspect pending queues, fallback/spill counts, invalid records and
   accumulation overflows. Identify expected CPU-completed processes rather
   than requiring every fallback counter to be zero. Report failed or
   excluded events and their reason; do not silently compare only successful
   survivors.
3. **Compare physical observables.** Use energy-ledger terms where coverage
   permits; longitudinal particle and deposited-energy profiles; observation
   particle multiplicities, energy and spatial distributions; and radio
   waveforms or spectra. Include distributions of derived quantities, such
   as the depth of a profile maximum, when relevant. Matching one integrated
   total does not constrain profile shape, tails or waveform timing.
4. **Quantify sampling uncertainty.** Use showers as the sampling units.
   For independent reference/candidate samples, the estimated standard error
   of a mean difference is
   :math:`\sqrt{s_C^2/n_C+s_R^2/n_R}`. For a justified paired design, estimate
   the uncertainty from the within-pair differences instead. State how pairs
   were constructed; a shared seed label is insufficient. Preserve bin and
   antenna correlations when resampling whole showers or using covariance
   matrices. Thinned weights do not make profile sums independent Poisson
   counts. Check the assumptions of the chosen uncertainty method.
5. **Apply the declared acceptance rule.** Report differences with confidence
   intervals, distributions and effect sizes, including tails where needed.
   To claim equivalence, require the uncertainty interval to fit within the
   predeclared equivalence bounds. A test that does not reject a difference
   is not evidence that an arbitrarily small difference has been excluded.
   Account for multiple bins, observers and observables in the test design;
   distinguish exploratory diagnostics from the acceptance criteria.

Profile comparisons must use the same depth grid and crossing convention.
Deposited energy in ``energyloss/dEdX.parquet`` is GeV **per bin**; division
by bin width is required for a density per unit slant depth. Retain particle
weights and the distinction between production counts, transported-particle
crossings and observation records.

Radio comparisons must retain signed components on compatible time grids.
The standard accumulator returns CoREAS electric field in V/m and ZHS vector
potential in V s/m; compare fields only after the defined ZHS differentiation
and endpoint convention. State any filtering, interpolation or time shift,
and report timing offsets before removing them. Compare per-observer peak,
norm and waveform residuals as appropriate; a small aggregate norm can hide
a discrepancy at a weak observer. Combine coherent fields before computing
squared-field quantities such as fluence. Equal fluence alone does not
establish equal waveform phase, polarization or arrival time. See
:doc:`../radio/accumulation` and :doc:`../radio/propagation`.

.. _c8_air_shower.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp
.. _PositronAtRest.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/utility/PositronAtRest.hpp
.. _testStochasticContracts.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/stochastic/testStochasticContracts.cpp
.. _ContinuousChecks.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/continuous/ContinuousChecks.hpp
.. _continuous/CMakeLists.txt: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/continuous/CMakeLists.txt
.. _testPhotonCascade.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/testPhotonCascade.cpp
.. _testLeptonCascade.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/testLeptonCascade.cpp
.. _testProfileAccumulation.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/output/testProfileAccumulation.cpp
.. _testLeptonStatistics.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/testLeptonStatistics.cpp
.. _testMaterialBoundaryStatistics.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/testMaterialBoundaryStatistics.cpp
.. _testRadioProjection.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/testRadioProjection.cpp
.. _testRadioAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testRadioAccumulator.cpp
.. _testFullWaveAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/firn/testFullWaveAccumulator.cpp
.. _testPhotonFinalState.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/finalstate/testPhotonFinalState.cpp
.. _Philox.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/random/Philox.hpp
.. _Philox.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/random/Philox.inl
.. _AirShowerCoordinator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp
.. _StaticFrontierIO.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/StaticFrontierIO.cpp
.. _WorkerSession.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/WorkerSession.inl
.. _WorkerSeedTest.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/coordinator/WorkerSeedTest.cpp
.. _WorkerSessionTest.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/coordinator/WorkerSessionTest.cpp
.. _testDecisionReplayTape.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/diagnostics/testDecisionReplayTape.cpp
.. _ShowerReport.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/ShowerReport.inl
.. _ProfileResult.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/output/ProfileResult.hpp
.. _compare.py: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/coordinator/compare.py
