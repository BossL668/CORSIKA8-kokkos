Run a first shower
==================

Start with a small event to check the executable, model data and output paths.
The example below uses a 10 GeV photon and eight radio observers. It is an
installation check; its particle population is too small to assess GPU
throughput, and a small radio signal is not a validation failure by itself.

Set the runtime paths
---------------------

Choose the installation you built. For OpenMP:

.. code-block:: bash

   export C8_INSTALL="$HOME/corsika8-workspace/install/openmp"
   export C8_BIN="$C8_INSTALL/bin/c8_air_shower"
   export CORSIKA_DATA="$C8_INSTALL/share/corsika/data"
   mkdir -p "$HOME/corsika8-runs"
   cd "$HOME/corsika8-runs"
   "$C8_BIN" --help

Use ``install/cuda`` or ``install/hip`` for a GPU executable. Retain the
compatible runtime environment and ``FLUPRO`` from installation. For HIP,
source the site's ROCm/DTK environment and check the installed device helper:

.. code-block:: bash

   "$C8_INSTALL/bin/c8_hip_devices"

The output directory supplied by ``--filename`` must not already exist.
Choose a new name for each run; do not create that directory in advance.

Select one execution path
-------------------------

**Native C8 CPU path**

.. code-block:: bash

   "$C8_BIN" --pdg 22 --energy 10 --nevent 1 --seed 18641 \
     --zenith 0 --azimuth 0 --emthin 0 --ring 1 \
     --em-backend proposal --radio-backend cpu \
     --filename first-native > first-native.log 2>&1

**Kokkos OpenMP** (with the OpenMP executable)

.. code-block:: bash

   "$C8_BIN" --pdg 22 --energy 10 --nevent 1 --seed 18641 \
     --zenith 0 --azimuth 0 --emthin 0 --ring 1 \
     --em-backend kokkos-proposal --radio-backend kokkos \
     --kokkos-execution openmp --kokkos-num-threads 4 \
     --filename first-openmp > first-openmp.log 2>&1

**Kokkos GPU** (with the matching CUDA or HIP executable)

For example, after setting ``C8_INSTALL``, ``C8_BIN`` and ``CORSIKA_DATA`` to the
CUDA installation:

.. code-block:: bash

   "$C8_BIN" --pdg 22 --energy 10 --nevent 1 --seed 18641 \
     --zenith 0 --azimuth 0 --emthin 0 --ring 1 \
     --em-backend kokkos-proposal --radio-backend kokkos \
     --kokkos-execution cuda --device 0 --gpu-memory-fraction 0.70 \
     --filename first-cuda > first-cuda.log 2>&1

For HIP, use its executable, replace ``cuda`` with ``hip``, and choose a new
output/log name. The execution space must be compiled into the executable.
In this air-shower application, ``kokkos-proposal`` requires the Kokkos radio
backend as well.

The memory fraction applies to device memory that is free at initialization.
It is a budget, not a request for a fixed allocation or a target for GPU
utilization. Small particle batches can still take CPU expansion steps.

Inspect completion
------------------

Immediately after the command finishes, check its exit status and log:

.. code-block:: bash

   echo "$?"  # zero means the process reported success
   tail -n 30 first-openmp.log
   cat first-openmp/summary.yaml

Use the name of the run you selected. A completed direct run has a final
``summary.yaml`` with the expected shower count. For Kokkos PROPOSAL, also
inspect ``gpu_em/summary.yaml`` for the event completion status. The
:doc:`outputs` page describes the data and explains why an existing output
directory alone does not prove completion.

The first run may spend time initializing models and generating or loading
PROPOSAL tables. These tables are prepared by the program; a separately copied
physics-table file is not required for this workflow.

Increase the event count
------------------------

Use ``--nevent 20`` (or ``-N 20``) with a new output name to run multiple showers
in one process. This retains model initialization between events. The seed
initializes the run's random streams; it does not request twenty identical
showers. Changing execution mode, thread count or scheduling can change
individual histories, so physics comparisons require suitable numerical and
statistical checks.

Next, use :doc:`configuration` to choose the primary, models and observing
geometry. The examples above follow the `air-shower entry point
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/applications/c8_air_shower.cpp>`_
and `backend option validation
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/BackendCli.cpp>`_.
