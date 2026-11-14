.. _the-hybrid-event-loop:

Hybrid event execution
======================

``HybridCascade`` advances an event through the native C8 scalar stepper and
the EM backend. Its host-side event loop selects particles from the C8 stack,
routes supported states through ``EmRouter`` and continues until all work is
complete. Kokkos kernels execute within the backend calls.

.. _construct-the-event-s-scheduling-objects:

Scheduling-object construction
------------------------------

The :ref:`application handoff <application-source-flow>` ends in
``run_hybrid_cascade(...)`` in ``HybridCascadeRunner.inl``. It receives native
C8 objects, an environment snapshot, transport geometry and callbacks:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Input
     - Responsibility
   * - ``environment``, ``tracking``, ``sequence``, ``output``, ``stack``
     - References used by ``HybridCascade`` and its native scalar stepper.
   * - ``environmentSnapshot``, ``geometry``, ``coordinateSystem``
     - Borrowed numerical environment data and native node mapping used by
       particle conversion, routing and CPU return paths.
   * - ``backendFactory``
     - The application's ``begin_backend`` callback. Initializes or reuses the
       EM session and returns a backend reference.
   * - ``fallbackFactory`` / ``outputSinkFactory``
     - Create the CPU completion handler and C8 output adapter for this event.
   * - ``configureCascade`` / ``onComplete``
     - Apply run controls before execution; collect the final report after it.

The function validates the actual C8 sequence against the process registry,
calls ``backendFactory`` to obtain the backend, and constructs the fallback
handler, output sink and router. These objects are then connected to the event:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/HybridCascadeRunner.inl
   :language: cpp
   :start-at: using tracking_type = std::remove_reference_t<TTracking>;
   :end-at: fallbackHandler);
   :lineno-match:

The function returns ``void``. Particle state is updated through the stack and
backend, while records reach the application's writers through the output sink.
``onComplete`` receives the backend, cascade, router, sink and fallback handler
by reference. The event-local objects remain alive until that callback returns;
the backend session can persist across events. The native environment and its
snapshot must remain valid throughout execution.

.. _acquire-and-route-a-particle:

Particle selection and routing
------------------------------

``HybridCascade::run()`` starts the output event, sets environment nodes and
enters the scheduling loop. ``CpuParticleScheduler::acquireNext()`` searches the
stack from its end for a non-erased, non-suspended particle. It returns a
``ScheduledParticle`` containing the particle handle, history and parent IDs,
generation, and the step ID from ``beginTransportStep()``.

An explicitly forced interaction or decay is resolved on the scalar path
before ordinary routing. Otherwise, ``EmRouter::canRoute(particle, stepId)``
checks CPU-return markers, particle type, backend capability and environment
support. When routing is accepted, the loop transfers the particle as follows:

.. literalinclude:: ../../../corsika/detail/kokkos/scheduling/HybridCascade.inl
   :language: cpp
   :start-at: emRouter_->stage(scheduled.particle_, scheduled.historyId_,
   :end-at: advanceInterleavedRouter();
   :lineno-match:

``stage(...)`` converts the state to the numerical representation and stores
it in the host vector ``staged_``. The loop then erases the native stack entry,
transferring responsibility for advancing that state to the router.
``completeParticleStep()`` closes the host selection and staging operation.
The particle remains active in the router. The subsequent
``advanceInterleavedRouter()`` call can begin backend transport if the dispatch
condition is satisfied.

.. _advance-scalar-and-device-work:

Scalar and backend interleaving
-------------------------------

Particles not routed use ``ScalarCascadeStepper::advance(particle)`` with the
native environment, tracking and process sequence. Surviving particles and
secondaries remain available through the C8 stack. An optional hadronic process
pool can suspend a selected particle while an interaction request is outstanding;
this is separate from the CPU-prefix coordinator discussed later.

For routed work, ``readyForScalarInterleave()`` allows backend execution before
the scalar stack is exhausted. In the standard resident path, it becomes true
when the staging vector reaches ``minimumBatchSize()``, resident photons or
leptons remain pending, or ``forceSmallBatchToGpu_`` requests a dispatch.
After staging a particle, ``advanceInterleavedRouter()`` processes router work
while this condition holds. When the CPU scheduler has no runnable particle,
the event loop also processes pending router work. Returned CPU particles have
their environment nodes refreshed before subsequent scalar steps.

In this path, host control alternates between scalar processing and backend
calls. Kokkos parallelizes particles within the backend kernels; the control
flow described here does not establish concurrent scalar and GPU execution.
Particle histories persist across calls, while kernel execution policies
determine the thread assignment for each launch. The optional
independent-subshower path has separate readiness and completion rules.

.. _finish-only-after-all-work-is-consumed:

Completion and finalization
---------------------------

The loop checks the following sources of work:

* Runnable particles in ``CpuParticleScheduler``.
* Work retained by the EM router: staged particles, resident photon/lepton
  queues, deferred reaction completions and pending handling of their products.
* Deferred hadronic interactions managed by the optional process pool.

For coordinator workers, ``BufferedFrontierRouter`` also reports unread
frontier roots as pending work. Cascade-equation hooks run inside the outer
loop because they can add particles to the stack.

At completion, ``EmRouter::endOfShower()`` checks that its queues are drained,
downloads enabled accumulators and passes results to the output sink.
``output_.endOfShower()`` closes the shower. Only then does the runner execute
its reporting callback. Emptying the native stack alone is not a completion
criterion.

The :doc:`batches` page describes backend arguments, returned records and the
handling of CPU reaction completions and small-batch scalar steps.
