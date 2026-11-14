Performance measurement
=======================

Performance results need a workload, a timing boundary and an accounting unit.
A complete-shower wall time, an accumulated host-call duration and a device
kernel duration answer different questions. This page defines the measurements
available at revision ``d5297c43c041a22b43c65c322d9b51bc7ec8e397`` and a procedure
for comparing execution configurations. It contains no measured speedups.

Timing boundaries and output files
----------------------------------

The direct air application in `c8_air_shower.cpp`_ builds the environment,
models, observers and execution session before entering its shower loop.
It has no timer spanning entry to and exit from ``main``. Measure complete
process wall time with an external recorder, retaining the exit status and
including all child processes in the completion boundary. This includes
startup, model and table preparation, shower execution, final output and
shutdown. The root ``summary.yaml`` contains a formatted ``runtime`` from
``OutputManager`` construction, or reopening for a resident worker; it is
not a monotonic timer for the full process lifetime.

The following fields have narrower, explicit scopes:

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Output and field
     - Boundary and interpretation
   * - ``simulation_timing/summary.yaml``:
       ``shower_<id>.wall_time_ms``
     - Elapsed ``steady_clock`` milliseconds between the timing output's
       ``startOfShower`` and ``endOfShower`` callbacks. The
       ``simulation_timing`` registration follows the physics writers in
       the ordered output map, including their preceding end-of-shower
       flushes. It excludes model setup and work after the cascade returns,
       such as interaction-histogram output and library finalization.
   * - ``gpu_em/summary.yaml``:
       ``hybrid_timing_ms.total_run``
     - Host elapsed time for ``HybridCascade::run``. It brackets all
       ``startOfShower`` callbacks, transport, router finalization and
       ``endOfShower`` callbacks. It is slightly broader than the timing
       output's callback interval.
   * - ``gpu_em/summary.yaml``:
       ``hybrid_timing_ms`` phase entries
     - Host durations for output callbacks, node assignment, routing,
       scalar stepping, stack processes, router advancement and cascade
       equations. They diagnose the hybrid loop; nested router and scalar
       diagnostics must not be added again to this total.
   * - ``gpu_em/summary.yaml``:
       ``backend_lifecycle.one_time_initialization_ms``
     - ``Kokkos::Timer`` around backend ``initialize``. Initialization fences
       its execution instance before stopping the timer. This excludes
       preparation performed before that call and is not total process
       startup time. The value is retained across reused showers; count it
       once per initialized backend, not once per output record.
   * - ``gpu_em/summary.yaml``:
       ``photon_backend_wall_time_ms``, ``lepton_backend_wall_time_ms``
     - Accumulated host-call durations for resident backend work, including
       the callback durations supplied by independent subshower drivers.
       They can include allocation, copies, launch overhead and waits.
       They are not device-only kernel durations.
   * - ``gpu_em/summary.yaml``:
       ``photon_host_postprocess_time_ms``,
       ``lepton_host_postprocess_time_ms``
     - Host time consuming returned records. These are components of the
       surrounding routing work, not additional event wall time.

The callback's success path records the time before flushing its own summary:

.. literalinclude:: ../../../corsika/detail/output/SimulationTiming.inl
   :language: cpp
   :lines: 23-38
   :lineno-match:

Require ``closed: true`` and ``status: closed`` before using this interval as
a completed timing sample. ``invalid_callback_order`` and
``library_ended_during_shower`` identify invalid or interrupted scopes.
Closure alone does not prove that the physics run completed: also inspect
the exit status, expected shower count and backend completion record as
described in :doc:`../usage/outputs`. ``OutputManager`` can close callbacks
during destruction after an unfinished shower.

Coordinated runs
----------------

`AirShowerCoordinator.cpp`_ records seconds using ``steady_clock``. Preserve
the per-event timing records as well as the campaign summary; concurrent
worker intervals do not form a serial sum.

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Record
     - Meaning
   * - Event ``TIMING.json``: ``wall_s``
     - Event coordinator scope from its process manager's creation through
       prefix execution, partitioning, workers, validation and merge, up
       to recording completion. It excludes the earlier device discovery
       and some final marker-writing work.
   * - Event ``TIMING.json``: ``prefix_phase_wall_s``, ``merge_s``
     - Host intervals around the prefix phase and output merge respectively.
       They are contained in event wall time.
   * - Event ``TIMING.json``: ``parts``; child ``PROCESS_EXIT.json``
     - For a newly launched child, ``start_s`` and ``end_s`` are offsets
       from its process manager's start, and ``wall_s`` is elapsed since
       launch. The coordinator records exit when polling observes it, so
       these include polling granularity. They are not CPU seconds.
   * - Resident-worker event ``PROCESS_EXIT.json``
     - ``wall_s`` covers processing a request, including its output closure
       and library-summary flush. ``worker_session_start_s`` and
       ``worker_session_end_s`` use that worker's own clock origin.
       ``timing_scope`` explicitly identifies request processing.
   * - ``worker_pool/TIMING.json``
     - Process-lifetime records for reused workers, including initialization
       and inter-request waiting. They must not be charged in full to every
       request or added to the campaign's elapsed wall time.
   * - Campaign ``TIMING.json``: ``wall_s``, ``events``
     - Elapsed campaign interval after device discovery, through event
       execution and resident-pool shutdown, with individual event records.

In cooperative execution, ``accelerator.cooperative`` contains host timing
and scheduling diagnostics. ``endpoint_window_overlap_ms`` is the
intersection of independent host-driver call windows. Likewise,
``openmp_while_cuda_pending_ms`` and ``slices_while_cuda_pending`` describe
CPU work while a CUDA job is pending. A pending job can be waiting, copying
or computing. These fields do not measure simultaneous GPU-kernel and CPU
execution. ``coordinator_idle_wait_ms`` and
``cuda_result_service_delay_ms`` describe coordination costs, not hardware
utilization. The contracts are explicit in `CooperativeStatistics.hpp`_.

Instrumentation that is active at this revision
-----------------------------------------------

``--cpu-detailed-step-timing`` enables the scalar stepper's conditional
``steady_clock`` probes. The resulting
``scalar_phase_timing_by_pdg_ms`` map in ``gpu_em/summary.yaml`` includes
``cross_section_ms``, ``distance_sampling_ms``, ``tracking_ms``,
``continuous_limit_ms``, ``continuous_process_ms``, geometry phases,
``interaction_ms``, ``decay_ms`` and ``secondary_processing_ms``, alongside
counts. These are nested within scalar stepping and add clock-read and
bookkeeping overhead. Compare production throughput with this option off;
use a separate instrumented run to attribute cost. See `BackendCli.cpp`_
and `ScalarCascadeStepper.inl`_.

``--gpu-detailed-stage-timing`` is registered and propagated into backend
configuration, but the PROPOSAL execution implementation at this revision
does not populate ``LeptonPipelineTimingStatistics``. The production
definitions, execution implementation, report writer and cooperative merge
must be read together: a field's presence in a YAML report is not evidence
that a timer ran. The following PROPOSAL timing fields are currently
unpopulated and must be treated as unavailable measurements, even when
their serialized value is zero:

* ``lepton_pipeline_timing`` durations and its ``wavefronts`` count;
* top-level ``kernel_time_ms`` and ``transfer_time_ms``;
* the duration and operation fields in ``transfer_timing`` and
  ``synchronization_timing``;
* ``profile.kernel_time_ms``, ``profile.transfer_time_ms``,
  ``radio.kernel_launch_time_ms`` and ``radio.transfer_time_ms``.

The ``timing_schema_version: 2`` and ``timing_semantics`` strings in
`ShowerReport.inl`_ retain descriptions of CUDA-event sums and selected
host-copy timings. They describe the reporting schema, not an active
measurement in the current PROPOSAL path. Neither the requested CLI option
nor a zero-valued duration establishes timer coverage. In particular,
``gpuProfileEnabled()`` controls longitudinal-profile accumulation; it is
not a performance-profiler switch. See `BackendTimingStatistics.hpp`_,
`BackendInstanceImplementation.inl`_ and `CooperativeBackendMerge.inl`_.

The EGS4 adapter has a separate, active device timer. `Egs4C8DeviceRadio.inl`_
records CUDA or HIP events on the execution stream around track export and
radio accumulation, synchronizes the stop event, and adds the elapsed
interval to ``RunStatistics::radioKernelMs_``. The air application exports
it as ``native_egs4/summary.yaml``: ``radio_kernel_ms``, together with
``radio_projection_batches`` and ``radio_execution_space``. This interval
covers the wrapped device region, which can contain several kernel launches;
it is not the duration of one radio kernel or the whole shower. OpenMP uses
a fence in this wrapper and does not populate the device-event duration.
``radioProjectionWallMs_`` and queue wall-time fields exist in the EGS4
statistics interface but are not exported by this air application's report.

GPU traces, launch duration and throughput
------------------------------------------

There is no standalone GPU performance-profiler executable in this source
baseline. To measure the uninstrumented PROPOSAL kernels, collect an external
device timeline and retain its tool version, collection settings, trace,
device identity, stream identifiers and lost-record diagnostics. Host launch
API duration measures submission work; it does not establish when an
asynchronous kernel finishes. A host timer around a launch measures completed
device work only when its boundary also waits for that work, and such a wait
can change overlap. The Kokkos documentation specifies potentially
asynchronous `parallel_for dispatch
<https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html>`_
and the completion guarantee supplied by an `execution-space fence
<https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/fence.html>`_.

For device intervals :math:`[s_i,e_i]`, the accumulated duration
:math:`\sum_i(e_i-s_i)` counts every interval. Overlapping kernels or streams
can therefore contribute more time than the elapsed window. Even the union
of these intervals measures only covered activity, not occupancy, arithmetic
throughput or utilization of every device unit. Report kernel and copy
intervals separately and retain their overlap; do not add them to shower
wall time. A device-event interval around several launches is a region
duration, whereas a profiler's per-kernel activity record can describe an
individual launch.

For a specified measured region, report both work and time:

.. math::

   \begin{aligned}
   R_{\mathrm{steps}} &= \frac{N_{\mathrm{completed\ steps}}}{T},\\[0.6em]
   c_{\mathrm{step}} &= \frac{T}{N_{\mathrm{completed\ steps}}}.
   \end{aligned}

The first is throughput, and the second is amortized elapsed cost per
completed particle step. Neither is the latency experienced by one particle:
many particles execute concurrently, individual particles take different
paths, and a resident call can advance several wavefronts. Report the
distribution of per-launch durations separately if launch cost is the
question. Avoid calling either quantity simply "particle time".

Use a denominator that matches the region. ``particles_staged``,
``maximum_input_batch`` and ``wavefronts`` describe different work counts;
none is automatically a count of unique particles or completed physical
steps. The historical ``gpu_particles`` field sums the router's advanced
photon and lepton records, including the applicable execution endpoints; its
name is not proof of GPU placement. Cooperative ``adaptive`` endpoint
``transport_records`` counters count completed physical steps, while job
input histograms can count a continuing particle repeatedly. Cite the
counter's producer in `EmRouter.inl`_ or `CooperativeStatistics.hpp`_ when
normalizing a result.

Resource accounting
-------------------

Keep resource cost separate from elapsed time. For an explicitly bounded
process tree and allocation interval, define:

.. math::

   \begin{aligned}
   H_{\mathrm{CPU,used}} &=
       \frac{\sum_p(t_{p,\mathrm{user}}+t_{p,\mathrm{system}})}{3600},\\[0.6em]
   H_{\mathrm{CPU,allocated}} &=
       \frac{\int C_{\mathrm{allocated}}(t)\,dt}{3600}.
   \end{aligned}

.. math::

   H_{\mathrm{GPU,reserved}} =
       \frac{\int G_{\mathrm{reserved}}(t)\,dt}{3600}.

Here the times are seconds, :math:`C` is the allocated CPU-core count and
:math:`G` is the reserved GPU-card count. For a constant allocation these
reduce to core count or card count multiplied by elapsed hours. State whether
CPU allocation means physical cores or logical CPUs. Actual CPU time sums
scheduled user and system execution across threads; count each process once
and avoid adding a parent's already aggregated child CPU time again.

The coordinator records elapsed wall time, not process CPU usage. Its worker
count or ``accelerator.host_threads`` cannot substitute for CPU seconds.
Use a process-tree resource recorder or scheduler accounting for actual CPU
usage and reserved resources. GPU event durations are neither reserved
card-hours nor a measurement of resource utilization. Multiple devices and
overlapping streams require separate identity and scope information.

Memory fields also have limited scope. ``peak_device_bytes`` is the maximum
accounted retained backend storage observed by
``refreshDeviceMemoryStatistics``; ``workspace_bytes`` covers the photon
and lepton workspaces. ``automatic_resident_capacity`` contains budget and
planning estimates. They are not a driver-wide memory high-water mark or
host RSS, and on OpenMP the execution-space storage is host memory. Keep
external host RSS and device-memory measurements separately, recording the
sampling interval and whether a peak is sampled or continuously accounted.
Transfer-byte counters describe their instrumented data paths; they do not
measure memory bandwidth without a matching elapsed interval.

Complete-shower comparisons
---------------------------

Use a real-shower run when the question concerns event throughput, startup
cost, output cost or resources needed for a physics campaign. Follow
:doc:`../usage/first-shower` for the runnable application and
:doc:`../usage/configuration` for the physical settings. A comparison record
should preserve the following inputs and outputs:

1. Fix the source revision, binary/build configuration, model-data revision,
   hardware, selected devices, CPU affinity and thread settings. Record
   competing workloads and relevant clock/power settings.
2. Match the primary and energy convention, angles, atmosphere, field,
   interaction models, cuts, thinning, observation level, profile policy,
   radio observers and output configuration. Use a declared seed ensemble.
   A shared seed alone does not require identical histories across schedulers.
3. Save the command, ``config.yaml``, log, exit status, physics outputs,
   ``simulation_timing`` and backend summaries. For coordination, also save
   ``CONFIG.json``, ``TIMING.json``, the worker records and completion markers.
4. State whether table and auxiliary caches were warm, whether the process
   was new, and whether ``-N`` reused initialized state. Preserve
   ``backend_lifecycle`` and the ``proposal_native`` cache-hit metadata where
   available. Do not infer all startup cost by subtracting shower time from
   process time: the remainder also contains output and shutdown work.
5. Repeat runs and report the number of samples, central value and spread.
   Keep process wall time, per-shower time and allocation cost in separate
   columns. Compare matched timing scopes and identify failed or incomplete
   runs instead of averaging their partial times into completed samples.
6. Run the relevant observable and completion checks before accepting a
   speedup. Record changes in step counts, fallback counts, batch sizes and
   output volume so reduced or altered work is visible. See :doc:`numerics`
   for energy-accounting scopes and ensemble comparisons.

Detailed timing, full-step records, replay capture and an external profiler
can change output volume, memory traffic, synchronization or scheduling.
Keep an instrumented attribution run separate from the production throughput
sample and compare their observables. Quantify instrumentation overhead with
matched configurations when it matters to the conclusion.

Frozen-input module comparisons
-------------------------------

A frozen-input experiment isolates an algorithm or execution stage from
shower evolution. For example, charged-lepton transport consumes
``InteractionRecord`` values plus fixed physics-table, environment and
geometry views, and produces ``LeptonTransportRecord`` values and fallback
results. Radio accumulation consumes fixed track records and observer/
propagation configuration and produces waveforms and counters. These are
different workloads from a full shower with changing populations and CPU
feedback. See :doc:`../transport/lepton-step` and :doc:`../radio/accumulation`.

Preserve the input arrays or their reproducible generator, hashes, sizes,
PID/energy/material distributions, random keys, tables and configuration.
Use the same inputs for each candidate. Validate output statuses, records,
fallback provenance and input immutability before interpreting elapsed time.
`testLeptonTransport.cpp`_ and `BackendStages.cpp`_ provide examples of fixed
record comparisons and real-table stage calls; they are correctness tests,
not a supplied throughput benchmark or a set of performance results.

In a dedicated timing harness, declare whether allocation, initialization,
upload, execution, synchronization and download are inside the measured
boundary. Reuse buffers for a resident-compute experiment; include transfers
when measuring the complete stage API. Reset mutable inputs and outputs as
required between repetitions so later repetitions do the same work. Warm-up
and compilation costs belong in a separately identified interval. Retain
the batch-size distribution and completed-record count, and sweep relevant
sizes rather than extrapolating a large uniform batch to a whole shower.

Source map
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Source
     - Measurement responsibility
   * - `c8_air_shower.cpp`_; `OutputManager.inl`_
     - Setup, event loop, output ordering and library-runtime metadata.
   * - `SimulationTiming.hpp`_; `SimulationTiming.inl`_
     - Shower callback boundary, units and closure status.
   * - `HybridCascade.inl`_; `ScalarCascadeStepper.inl`_
     - Active host phase timing and optional scalar detail.
   * - `BackendInstanceImplementation.inl`_
     - Backend initialization timer, retained-memory accounting and execution.
   * - `BackendTimingStatistics.hpp`_; `ShowerReport.inl`_
     - Timing field definitions and serialized names; serialization does not
       imply an active producer.
   * - `BackendCli.cpp`_
     - Registered detailed-timing controls.
   * - `EmRouter.inl`_; `CooperativeStatistics.hpp`_;
       `CooperativeBackendMerge.inl`_
     - Host-call timing, work counters, overlap contracts and aggregation.
   * - `AirShowerCoordinator.cpp`_; `WorkerSession.inl`_
     - Event/campaign, process and persistent-request scopes.
   * - `Egs4C8DeviceRadio.inl`_; `Egs4RunSession.inl`_
     - Separate EGS4 radio device events and exported report fields.

.. _c8_air_shower.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp
.. _OutputManager.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/output/OutputManager.inl
.. _SimulationTiming.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/output/SimulationTiming.hpp
.. _SimulationTiming.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/output/SimulationTiming.inl
.. _HybridCascade.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/HybridCascade.inl
.. _ScalarCascadeStepper.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/ScalarCascadeStepper.inl
.. _BackendInstanceImplementation.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl
.. _BackendTimingStatistics.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/BackendTimingStatistics.hpp
.. _ShowerReport.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/ShowerReport.inl
.. _BackendCli.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/BackendCli.cpp
.. _EmRouter.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/EmRouter.inl
.. _CooperativeStatistics.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/execution/CooperativeStatistics.hpp
.. _CooperativeBackendMerge.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/CooperativeBackendMerge.inl
.. _AirShowerCoordinator.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp
.. _WorkerSession.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/WorkerSession.inl
.. _Egs4C8DeviceRadio.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/egs4/Egs4C8DeviceRadio.inl
.. _Egs4RunSession.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/atmosphere/Egs4RunSession.inl
.. _testLeptonTransport.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/transport/testLeptonTransport.cpp
.. _BackendStages.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/backend/BackendStages.cpp
