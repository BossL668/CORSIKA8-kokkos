Overview
========

CORSIKA 8 with Kokkos adds parallel electromagnetic transport and radio
calculation to the CORSIKA 8 framework. The application constructs the
environment, physics processes and observers. Library interfaces connect these
objects to the selected execution path, while the application retains its
process sequence and event loop.

The native executable is ``c8_air_shower``. Its default electromagnetic backend
is the C8 CPU path using PROPOSAL; Kokkos execution is selected explicitly at
runtime. Building a CUDA or HIP executable alone does not select accelerated
transport.

What the execution paths provide
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Path
     - Role
   * - Native C8
     - Runs the existing CPU cascade with PROPOSAL and CPU radio modules.
   * - Kokkos OpenMP
     - Runs the Kokkos particle and radio kernels on CPU threads.
   * - Kokkos CUDA / HIP
     - Runs supported electromagnetic transport and radio calculations on a GPU.
       Hadronic models and unsupported device reactions retain CPU processing.
   * - Coordinated execution
     - Adds optional CPU-prefix parallelization and cooperation among multiple
       GPUs within one shower. GPU workers can be reused across events.

``kokkos-proposal`` uses PROPOSAL physics and exported native table data, with
interpolation and transport calculations implemented in the Kokkos library.
An optional ``kokkos-egs4`` path is also available in builds that enable EGS4.
Choosing EGS4 changes the electromagnetic implementation; it is a separate
physics configuration, not just a choice of processor.

Environment and application scope
---------------------------------

The library exports supported C8 geometry and medium models into device-accessible
snapshots. The C8 environment tree remains the source of geometry and material
information. A new shape or material law must have a corresponding export and
device implementation before it can be used on the GPU.

The :doc:`usage/index` starts with the atmospheric ``c8_air_shower`` application.
The repository also contains separate ice and firn cascade applications. Their
scene construction and radio propagation require their own configuration and
validation; the air-shower commands are not a general-purpose medium selector.

Where to start
--------------

* **Run the program:** follow :doc:`usage/installation` and
  :doc:`usage/first-shower`.
* **Set up a study:** read :doc:`usage/configuration` before changing models,
  observing geometry or transport settings.
* **Choose computing resources:** see :doc:`usage/parallel-execution`.
* **Check a result:** see :doc:`usage/outputs` for completion records, data and
  timing definitions.

For the implementation, continue with :doc:`application`: it traces the function
calls, arguments, ownership and results of the application interface. The remaining
architecture and maintenance chapters are being written. Implementation
availability, numerical validation and performance measurements are separate;
a build profile does not establish scientific validation on every device it
can target.

Source reference
----------------

This overview follows the `application and library layout at d5297c43
<https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
This repository and guide are not an official CORSIKA documentation release.
