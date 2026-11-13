Outputs and run checks
======================

Keep the log and configuration with the output. A directory can be created
before a shower finishes, so its existence is not a completion test.

Check a direct run
------------------

A successful direct run should have a zero exit status and a final
``summary.yaml`` reporting the requested number of showers. For
``kokkos-proposal``, inspect each event's ``complete`` and ``status`` fields in
``gpu_em/summary.yaml`` as well. A timing record marked ``closed`` records the
end of its timing scope; it does not independently establish physics validity.

The standard air application writes these output groups:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Location below the run directory
     - Content
   * - ``config.yaml``, ``summary.yaml``
     - Run metadata and final shower count.
   * - ``profile/profile.parquet``
     - Longitudinal particle profiles versus depth.
   * - ``energyloss/dEdX.parquet``
     - Energy deposited in depth bins.
   * - ``particles/particles.parquet``
     - Particles recorded at the observation level.
   * - ``production_profile/``
     - Particle production profile.
   * - ``CoREAS/``, ``ZHS/``
     - Observer configuration and time-domain radio data.
   * - ``gpu_em/``
     - Kokkos transport configuration, completion status and counters.
   * - ``simulation_timing/``
     - Per-shower wall timing and associated metadata.

Table rows carry a ``shower`` index. Inspect the schema and accompanying
configuration before combining events. Longitudinal ``X`` is in
:math:`\mathrm{g\,cm^{-2}}`; the deposited energy is stored in GeV per depth
bin. Divide by the bin width if a density per unit slant depth is required.

Radio samples for different observers are stored in their configured order.
Read the observer metadata before treating rows as a single waveform or
combining antennas. A valid observer window and consistent time sampling are
essential for waveform and fluence comparisons.

For a lightweight Parquet inspection, optionally install ``pyarrow`` in your
analysis environment. The following reads metadata rather than loading all
particle records:

.. code-block:: python

   from pathlib import Path
   import pyarrow.parquet as pq

   run = Path("first-openmp")
   for relative in ("profile/profile.parquet", "energyloss/dEdX.parquet"):
       table = pq.ParquetFile(run / relative)
       print(relative, table.metadata.num_rows, table.schema.names)

Check a coordinated run
-----------------------

The coordinator additionally writes ``CONFIG.json``, ``TIMING.json`` and
completion records. For multiple events, ``CAMPAIGN.json`` describes the
campaign and individual ``seed_*`` directories hold event results. Physics
outputs from the workers are collected under each event's ``merged`` directory.

Check that all requested events are present and marked complete. An
``INCOMPLETE.json`` record or a failed worker must be investigated even if
some output files exist. ``COMPLETE.json`` records process completion; it
is not a statistical acceptance report.

Interpret timing consistently
-----------------------------

Process wall time includes startup, model/table preparation, shower execution
and final output. The per-shower ``wall_time_ms`` in
``simulation_timing/summary.yaml`` covers the application's shower timing
scope. Coordinated runs also record stage timings in ``TIMING.json``.
Use the same timing boundary for every configuration in a comparison.

For ``-N > 1``, initialization is retained across events. Report startup and
event times separately when studying simulation throughput. Profiling and
replay options can add overhead, so record whether they were enabled.

Allocated GPU hours are GPU count multiplied by elapsed allocation time in
hours. They are different from summed GPU kernel activity. CPU core-hours
measured from process CPU time are likewise different from allocated CPU
cores multiplied by wall time. State which quantity is reported.

Common problems
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Check
   * - Output path is rejected
     - Choose a new ``--filename``. Do not reuse a partially written directory
       as though it were a fresh run.
   * - Model files cannot be found
     - Set ``CORSIKA_DATA`` to this backend's installed data directory and
       retain the complete FLUKA runtime installation.
   * - Requested backend is unavailable
     - Check that the executable was built for that execution space. Runtime
       flags do not compile missing backends or EGS4 support.
   * - HIP compiles but cannot run
     - Activate the matching ROCm/DTK environment and check device-file
       permissions with the administrator. Run ``c8_hip_devices``.
   * - No radio signal or missing observers
     - Check the antenna count, coordinate convention, observation geometry
       and recording window. A default run without observers has no array.
   * - Low GPU utilization in a small shower
     - Check particle counts and CPU work before interpreting utilization.
       Small-batch expansion and CPU reactions are part of the execution path.

Before a physics comparison
---------------------------

Match the primary, energy definition, angles, hadronic and electromagnetic
models, atmosphere, field, cuts, thinning, observation level, profile counting
mode and radio configuration. Keep the same model-data revision. A common
seed is useful but does not ensure identical histories across different
schedulers or backends.

Check particle profiles, energy deposition and radio observables using a
sample appropriate to the question. A successful installation check or a
single matching total is not sufficient to establish statistical agreement.

Source references
-----------------

See the `output manager
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/output/OutputManager.inl>`_,
`shower timing
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/output/SimulationTiming.inl>`_
and `coordinator output handling
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp>`_.
