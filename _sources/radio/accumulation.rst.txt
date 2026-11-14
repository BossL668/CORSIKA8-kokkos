Parallel projection and waveform accumulation
=============================================

``RadioAccumulator<TExecutionSpace>`` projects completed electron and positron
tracks onto observers and accumulates the corresponding CoREAS and ZHS signals.
The execution space selects the CPU or GPU implementation of the parallel work;
it does not select the emission model. This page describes the standard
accumulator at the source revision recorded in :doc:`../radio`.

The inputs are described in :doc:`inputs`. Firn geometrical-optics calculation
uses a separate accumulator and the ``TimeDomainEmission`` deposition helpers;
it must not be inferred from the standard accumulator's policies below. Its
propagation and emission path is described in :doc:`propagation`.

Accumulator interface and lifetime
----------------------------------

The accumulator owns its observer and propagation arrays, waveform buffers,
diagnostic counters, track workspace and host staging buffers. Input device
views remain borrowed: the caller must retain their storage until the queued
projection has finished. Operations belonging to one accumulator use the same
execution-space instance so that submissions and workspace reuse are ordered.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Interface
     - Responsibility
   * - ``initialize(config, tuning, execution)``
     - Validate configuration, upload immutable observer and propagation data,
       and allocate the waveform and diagnostic storage.
   * - ``accumulateLeptonTracks(records, count, execution)``
     - Submit the first ``count`` device records for projection. It neither
       transports particles nor returns a completed waveform.
   * - ``enqueueHostLeptonTracks(records, count, execution)``
     - Copy host-produced records into owned staging storage. Full staging
       buffers are uploaded and submitted to the same accumulator.
   * - ``flushHostLeptonTracks(execution)``
     - Upload and submit remaining host records. The upload is completed
       before its host mirror is reused.
   * - ``download(execution)``
     - Flush pending tracks, complete waveform retrieval and return owned
       ``RadioWaveforms`` with one three-component waveform per observer.
   * - ``downloadFixed(identity, endpoint, execution)``
     - Finalize a deterministic accumulator and return an owned integer
       snapshot for a compatible cooperative merge.
   * - ``reset(execution)``
     - Clear waveforms, counters and event statistics while retaining prepared
       configuration and allocated resources for another event.

Initialization is not a per-track operation. Workspace capacity can be reserved
with ``reserveTrackCapacity``; growing it may require a fence before the old
allocation is released. Normal submissions do not fence each transport
wavefront. ``reset`` rejects pending host records, clears device storage and
fences before returning. Event owners should finalize an event before resetting
and reusing its accumulator. See `RadioAccumulator.hpp`_ and
`RadioAccumulator.inl`_.

Track--observer work distribution
---------------------------------

One projection computes one track's contribution to one observer. Different
pairs can write the same observer's time bins, so pair-level parallelism also
requires a defined accumulation strategy.

The standard implementation selects between two submission paths:

.. list-table::
   :header-rows: 1
   :widths: 29 34 37

   * - Path
     - Selection
     - Work organization
   * - Direct projection
     - Non-OpenMP execution and fewer than 16,384 input tracks
     - A bounded ``RangePolicy`` visits track--observer pairs. Each pair
       constructs its track kinematics before evaluating the emission.
   * - Precomputed, tiled projection
     - OpenMP, or at least 16,384 input tracks on other execution spaces
     - A first kernel prepares each track once. A ``TeamPolicy`` then groups
       tracks and observers into tiles for projection.

The direct kernel obtains its indices from the pair index:

.. literalinclude:: ../../../corsika/detail/kokkos/radio/RadioAccumulator.inl
   :language: cpp
   :lines: 682-683
   :lineno-match:

This mapping describes logical Kokkos iterations, not a persistent assignment
of a particle to a GPU thread. Invalid track records return without emission.
The initial track-recording pass also records valid-track diagnostics once per
track, rather than once per observer.

In the tiled GPU path, a team cooperatively loads its tracks and observers into
team scratch storage, synchronizes, and evaluates the tile's pair combinations.
Track kinematics and observer records can therefore be reused within the team.
The default tile contains 8 tracks and 32 observers; the generic default team
size is 256. The number of pairs, tile dimensions and team size are distinct
parameters. Tail tiles retain bounds checks for unused lanes.

When both algorithms are enabled and their observer positions and time grids
match exactly, ``fuseCoreasZhs_`` selects a combined pair kernel. It calls both
``accumulate_co_reas`` and ``accumulate_zhs`` with shared pair inputs. It does
not replace either emission formula or combine their output buffers. Different
observer collections use separate projections. These paths are implemented in
`RadioAccumulator.inl`_ and `RadioProjection.inl`_.

Execution-space policies
-------------------------

**OpenMP.** The default tile is 4 tracks by 4 observers. A team of size one
evaluates a tile's pairs; different tiles execute on different OpenMP workers.
In deterministic mode, each worker has a private integer waveform row selected
by a Kokkos ``UniqueToken``. Updates to that row require no atomics. Finalization
merges the rows and checks their integer sums for overflow. Floating-point
accumulation uses the shared waveform path instead. The private-row operations
and merge kernel are defined in `RadioAccumulatorOperations.inl`_.

**CUDA and HIP.** Waveform bins are shared across teams and updated atomically.
For tiled ZHS work with at least one full observer tile, a specialized functor
first accumulates four integer diagnostics locally, then reduces them within
the team. These are CoREAS contributions, ZHS contributions, ZHS subtracks and
fixed-point overflows. This optimization reduces global counter updates;
**waveform samples still use atomic accumulation**.

For HIP ZHS kernels, a default team selection of 256 is changed to a team of
128 with ``Kokkos::LaunchBounds<128, 2>``. An explicit team size of 128 uses that
policy as well. The actual team size is capped by the number of pairs in
the tile, so an explicit size of 256 is retained only when the tile has at
least 256 pairs. Other cases use the generic team policy. These are launch and counter-management choices around
the same emission functions, not HIP-specific physics approximations. They
do not by themselves establish a performance improvement for every geometry
or observer count.

``RadioAtomicOperations`` supplies a narrow device-primitive adapter: CUDA and
HIP device compilation use native atomic operations, while host compilation
uses Kokkos atomics. Projection and emission remain in the common C++ kernels.
See `RadioAtomicOperations.inl`_.

Waveform representation and numerical contract
-----------------------------------------------

Each enabled model stores three component arrays with contiguous bin ranges
for its observers. CoREAS accumulates electric field in V/m. ZHS accumulates
vector potential in V s/m; differentiating it into electric field belongs to
the subsequent observer output stage, not ``RadioAccumulator::download``.

``RadioConfig::deterministic_`` selects signed 64-bit fixed-point accumulation
and defaults to true. For the configured field scale :math:`E_{\mathrm{scale}}`
and an observer sampling rate :math:`f_s`, the conversion factors are

.. math::

   S_{\mathrm{CoREAS}} = \frac{2^{62}}{E_{\mathrm{scale}}},
   \qquad
   S_{\mathrm{ZHS}} = \frac{2^{62} f_s}{E_{\mathrm{scale}}}.

A contribution is scaled and rounded before integer addition. The configured
field scale controls this representation; it is not an amplitude cut.
Non-finite or unrepresentable increments and overflowing sums are recorded.
Finalization rejects an overflowed accumulator instead of silently returning
clipped or wrapped waveforms. Increasing the field scale gives more amplitude
headroom at the expense of a coarser quantization step.

Integer addition makes the sum independent of addition order when the
individual quantized contributions are identical and no overflow occurs.
It does not guarantee bitwise equality across CPU and GPU architectures:
different floating-point evaluation of a contribution can change its rounded
integer. Floating mode uses double-precision additions and retains their
order dependence. Floating diagnostic sums are also separate from the integer
waveform contract. The conversion and sample-deposition rules are in
`RadioProjection.inl`_.

Finalization and cooperative merging
------------------------------------

``download`` flushes host records, merges OpenMP private rows if present, and
retrieves counters and waveforms through synchronous host mirror copies. It
checks overflow and decodes each observer's samples into ``RadioWaveforms``.
Retrieving the same completed result twice is rejected. Use ``reset`` to start
the next event with retained allocations.

``downloadFixed`` instead copies integer buffers and counters on the supplied
execution instance and explicitly fences before exposing the host snapshot.
It seals the accumulator before exporting, preventing repeated folding of
private rows or additional submissions after a failed export. The snapshot
includes shower identity, endpoint identity, configuration and owned integer
arrays; its ZHS array is still vector potential.

``CooperativeRadioMerge`` handles the two endpoints defined by
``CooperativeEndpoint``: ``Cuda`` and ``OpenMP``. It is **not** the general
multi-GPU process-output merger. ``commit(snapshot)`` checks shower identity,
endpoint uniqueness, exact compatible configuration and overflow before
accepting a snapshot. A failed commit preserves the previous accumulated
state. ``take()`` requires both endpoints and can be called once.

The merge adds integer samples before ``decode_fixed_radio`` converts them
to floating-point waveforms. Observer grids, propagation configuration,
quantization scale and enabled algorithms must agree. This avoids merging
unrelated time bins or differently scaled integers. See
`FixedRadioSnapshot.hpp`_ and `CooperativeRadioMerge.inl`_. The process
coordinator's event-output merging is described in
:doc:`../execution/coordinator`.

Counters and timing interpretation
----------------------------------

``RadioStatistics`` distinguishes input records, valid tracks, projected pairs,
deposited contributions, ZHS subtracks, tiles and transfer bytes. In particular:

* ``trackObserverPairs_`` counts submitted pair evaluations for each enabled
  algorithm, including pairs that later produce no in-window contribution.
* ``fusedTrackObserverPairs_`` counts shared pair work when both models are
  projected together; it is not another set of physical tracks.
* Valid-track and contribution counters are copied from device storage during
  finalization. Intermediate host statistics are not a completed-event report.
* Contribution counts describe deposition operations, not a count of distinct
  particles, antennas or kernel launches.

The structure also declares device, kernel and transfer timing fields. The
standard ``RadioAccumulator`` does not populate those elapsed-time fields at
this revision. Their presence in a report must not be interpreted as a measured
kernel interval. Performance measurements require instrumentation around the
actual execution and completion boundaries, separating asynchronous submission
time from completed device work. See `RadioStatistics.hpp`_.

Validation coverage and maintenance boundaries
----------------------------------------------

The following source tests cover accumulator contracts. Their presence is not
a claim that a particular device or build has passed them.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Test source
     - Coverage
   * - `testRadioAccumulator.cpp`_
     - Direct and tiled submissions, fixed and floating accumulation, individual
       and combined models, capacity boundaries, and default versus explicit
       HIP tile policies against retained reference implementations.
   * - `testRadioCounters.cpp`_
     - Counter-reduction equivalence with direct projection, including invalid
       records, partial tiles and separate observer collections.
   * - `testRadioMerge.cpp`_
     - Endpoint order, identity and configuration checks, integer overflow,
       failed-commit preservation and explicit nonblocking GPU stream export.
   * - `testHostRadioStaging.cpp`_
     - Host-record staging and queued-producer behavior.
   * - `testHomogeneousRadio.cpp`_
     - Homogeneous propagation and air-regression cases.

These checks support launch-policy and accumulation refactoring. They do not
replace independent emission-model validation or full-shower comparisons.
Native C8 CoREAS/ZHS and the standard Kokkos projection remain separate
implementations; equivalent model choices do not imply one shared native
function body. ``TimeDomainEmission`` provides common time-domain deposition
helpers for the firn geometrical-optics accumulator, rather than replacing
every native and standard Kokkos radio implementation.
The helpers accept ray timing and projected displacement: ``deposit_zhs``
deposits the track's vector-potential contribution, while ``deposit_coreas``
combines matched endpoint rays and handles the near-Cherenkov limit. Ray-family
selection and track subdivision remain the caller's responsibility. Their
interfaces are defined in `TimeDomainEmission.hpp`_.

.. _RadioAccumulator.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioAccumulator.hpp
.. _RadioAccumulator.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioAccumulator.inl
.. _RadioAccumulatorOperations.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioAccumulatorOperations.inl
.. _RadioAtomicOperations.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/RadioAtomicOperations.inl
.. _RadioProjection.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/algorithms/RadioProjection.inl
.. _RadioStatistics.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/RadioStatistics.hpp
.. _TimeDomainEmission.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/algorithms/TimeDomainEmission.hpp
.. _FixedRadioSnapshot.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/radio/FixedRadioSnapshot.hpp
.. _CooperativeRadioMerge.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/radio/CooperativeRadioMerge.inl
.. _testRadioAccumulator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testRadioAccumulator.cpp
.. _testRadioCounters.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testRadioCounters.cpp
.. _testRadioMerge.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testRadioMerge.cpp
.. _testHostRadioStaging.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testHostRadioStaging.cpp
.. _testHomogeneousRadio.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testHomogeneousRadio.cpp
