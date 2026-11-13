Execution and scheduling
========================

The application hands a configured event to ``run_hybrid_cascade(...)``.
This chapter follows the next steps: selecting particles from the C8 stack,
collecting supported particles into batches, advancing those batches and
returning unfinished work or output to the host. It then follows the optional
coordinator that divides a shower among processes and GPUs.

.. note::

   Source baseline: `d5297c43
   <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.
   The main flow below describes ``kokkos-proposal`` with the ordinary resident
   EM backend. Native C8 execution takes the separate ``Cascade::run()`` branch
   described in :doc:`application/event-execution`. EGS4 and optional cooperative
   CPU/device endpoints have additional paths; they are not interchangeable
   with this flow.

Two levels of scheduling
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Level
     - Responsible code
     - Unit of work
   * - Within one process
     - ``HybridCascade``, ``CpuParticleScheduler`` and ``EmRouter``
     - A C8 particle step or a batch of supported particles. The backend runs
       Kokkos kernels on the selected execution space.
   * - Across processes
     - ``coordinator::run(...)`` and its worker pool
     - A static set of subshower roots assigned to each process. Each GPU worker
       uses the same local cascade and router interfaces.

An OpenMP backend runs particle kernels on CPU threads. A CPU-prefix worker is
a separate process running a native C8 prefix. These are different forms of
parallelism. Likewise, one router call can advance several resident wavefronts;
it does not represent one kernel launch or one complete shower.

.. toctree::
   :maxdepth: 1

   execution/hybrid-loop
   execution/batches
   execution/coordinator

.. _execution-source-files:

Source file map
---------------

The trees below select the files used in this chapter. Public declarations are
under ``corsika/kokkos/``, template implementations under
``corsika/detail/kokkos/`` and compiled implementations under ``src/kokkos/``.
Click a filename to open the pinned source.

.. include:: execution/_file-map.txt

.. _execution-data-flow:

From the application handoff to particle execution
--------------------------------------------------

.. figure:: _static/execution-data-flow.svg
   :alt: File and function flow from the application runner to the CPU scheduler, EM router, resident backend, CPU fallback and output sink.
   :width: 100%
   :target: _static/execution-data-flow.svg

   Solid arrows show calls and transfers; dashed arrows show returned work.
   The loop continues until the CPU stack, router and deferred work are drained.
   Open the full-size SVG to follow its source links.
   :download:`SVG <_static/execution-data-flow.svg>` ·
   :download:`Editable DOT source <_static/execution-data-flow.dot>`.

``HybridCascadeRunner.inl`` constructs the router, fallback handler and output
sink for the event. The backend session and its allocated resources can outlive
this call. ``HybridCascade::run()`` advances the event and invokes the router;
the router calls the resident EM backend and consumes its results.
The :doc:`execution/hybrid-loop` page follows these calls in order.

Where this chapter hands off
----------------------------

* Particle conversion uses ``particle_transfer::to_device_state(...)`` and
  ``import_particle(...)``. The :doc:`particles` chapter will describe units,
  coordinate systems, state fields and identity in detail.
* ``run_photon_cascade(...)`` and ``run_lepton_cascade(...)`` are the entry
  points to the resident transport loops. The :doc:`transport` chapter will
  follow their numerical kernels.
* Transport records and CPU-produced tracks reach output writers and radio
  accumulators. The :doc:`radio` chapter will describe track projection and
  waveform accumulation.

Maintenance and existing checks
-------------------------------

The following source tests help locate the contracts affected by a scheduling
change. They are existing tests, not a claim that a new physics or performance
campaign was run for this documentation update.

.. include:: execution/_checks.txt
