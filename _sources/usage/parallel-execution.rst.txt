Parallel execution
==================

CPU threads, CPU-prefix workers and GPU workers serve different roles. Select
resources for the intended run before adjusting batching parameters.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Work distribution
   * - OpenMP
     - CPU threads execute Kokkos particle and radio kernels within a shower.
   * - Direct single GPU
     - One application process uses one GPU; the CPU prefix remains serial.
   * - Parallel CPU prefix
     - Separate CPU worker processes advance portions of the initial cascade
       before GPU handoff. The worker count is independent of GPU count.
   * - Multiple cooperating GPUs
     - GPU workers share one shower. ``--nevent`` runs successive showers in
       that mode; it does not assign one complete shower to each GPU.

OpenMP and a direct GPU run
---------------------------

Use the matching executable and runtime paths from :doc:`first-shower`.
For OpenMP, ``--kokkos-num-threads 8`` requests eight threads. Optional OpenMP
placement settings such as ``OMP_PLACES=cores`` and ``OMP_PROC_BIND=close``
control affinity according to the available CPU allocation; they do not
reserve cores from other users.

For a direct single-GPU run, specify ``--device 0`` and omit
``--prefix-workers``. For example, with a CUDA executable:

.. code-block:: bash

   "$C8_BIN" -A 56 -Z 26 --energy 1e5 --zenith 0 --azimuth 0 \
     --nevent 20 --seed 18641 --emthin 0 --ring 1 \
     --em-backend kokkos-proposal --radio-backend kokkos \
     --kokkos-execution cuda --device 0 --gpu-memory-fraction 0.90 \
     --gpu-min-batch 16384 --filename fe-single-gpu

This is an example configuration, not a recreation of a previous benchmark.
Select the required antenna file, cuts and environment for your study.

CPU-prefix parallelism and multiple GPUs
----------------------------------------

Selecting more than one GPU or explicitly supplying ``--prefix-workers``
selects coordinated execution before model initialization. To use four GPUs
and sixteen CPU-prefix workers:

.. code-block:: bash

   "$C8_BIN" -A 56 -Z 26 --energy 1e5 --zenith 0 --azimuth 0 \
     --nevent 20 --seed 18641 --emthin 0 --ring 1 \
     --em-backend kokkos-proposal --radio-backend kokkos \
     --kokkos-execution cuda --device 0,1,2,3 --prefix-workers 16 \
     --gpu-memory-fraction 0.90 --gpu-min-batch 16384 \
     --filename fe-four-gpu

For HIP, use the HIP executable, ``--kokkos-execution hip`` and the device
indices reported by ``c8_hip_devices``. A single GPU can also use a parallel
prefix, for example ``--device 0 --prefix-workers 4``. Allocate CPU resources
for those processes as well as the GPU workers and coordinator.

The coordinator uses fixed primary energy and a positive seed for these runs.
The prefix workflow described here uses ``kokkos-proposal``. It is not an
instruction to enable coordinated EGS4 execution.

GPU workers are persistent by default for coordinated ``kokkos-proposal``
runs with multiple events. Models and GPU sessions are retained, while event
inputs and outputs remain separate. GPU initialization begins with the first
real work request after the CPU prefix; early initialization is disabled.
The direct single-process event loop also retains initialization across
``-N > 1`` events.

If the goal is throughput across independent events, run separate application
processes on separate GPUs, each with its own seed and output directory.
This is different from the cooperating multi-GPU command above. Neither mode
is universally faster; particle population, prefix cost and radio workload
all affect the result.

Memory and batching controls
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Control
     - Meaning
   * - ``--gpu-memory-fraction``
     - Budget as a fraction of currently free device memory at initialization.
       Set it explicitly: direct execution defaults to 0.70, the coordinator
       to 0.50. The coordinator accepts values up to 0.90.
   * - ``--gpu-min-batch``
     - Particle-count threshold for GPU dispatch and small-batch checkpoints;
       default 16384. It does not set the GPU block size or a fixed input size.
   * - ``C8_STATIC_FRONTIER_IMPORT_BATCH``
     - Number of staged frontier particles imported per chunk by coordinated
       execution. This is distinct from the dispatch threshold.
   * - ``--prefix-split-energy``
     - CPU-prefix subdivision threshold. Zero selects the automatic rule
       based on primary energy and CPU-prefix worker count.

For a controlled scheduling study, the import size can be set for one run:

.. code-block:: bash

   export C8_STATIC_FRONTIER_IMPORT_BATCH=32768
   # Run the chosen coordinated command here, then restore the default:
   unset C8_STATIC_FRONTIER_IMPORT_BATCH

The accepted explicit import range is 1--65536 particles. Larger chunks or
higher dispatch thresholds are not guaranteed to improve total time. Compare
several events under the same physical conditions and record the resolved
settings. Scheduling changes may change random histories; statistical
consistency and performance are separate checks.

Source references
-----------------

See the `execution selector
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/ExecutionSelection.hpp>`_
and `air-shower coordinator
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp>`_.
