Device storage and state updates
================================

``ParticleSoA<TMemorySpace>`` owns the field arrays used during batch
execution. ``WavefrontQueue<TExecutionSpace>`` combines two such allocations
with transfer buffers and queue metadata. The execution space selects where
operations run; its memory space determines where these arrays are allocated.

Field arrays and kernel access
------------------------------

SoA means *structure of arrays*: each field has a Kokkos view indexed by
particle. Positions and directions use two-dimensional views with three
components. ``ParticleSoA::load(index)`` returns one complete ``ParticleState``;
``store(index, state)`` copies all its fields into that slot.

``rawDeviceView()`` returns a non-owning ``ParticleSoARawView``. The scalar
fields are pointers; the two vector fields also carry strides through
``RawView2D``. This preserves the owning views' layouts without assuming that
the three components are contiguous for every backend.

The raw view's load operation is included below:

.. literalinclude:: ../../../corsika/detail/kokkos/data/ParticleSoA.inl
   :language: cpp
   :lines: 12-31
   :lineno-match:

The raw view does not allocate memory, transfer values or extend allocation
lifetime. Its pointers must be accessible to the kernel and remain valid
until queued work completes. Reallocation invalidates the borrowed addresses;
a buffer swap also changes which allocation represents the current front.
Acquire the appropriate view for each front before launching its kernels.

Allocations use ``Kokkos::WithoutInitializing``. A valid active slot therefore
requires all fields to be written before they are read. ``size()`` identifies
the active prefix; ``capacity()`` describes allocated storage, not the number
of valid particle states.

Queue upload and download
-------------------------

`WavefrontQueue.inl`_ implements the host/backend transfer:

.. list-table::
   :header-rows: 1
   :widths: 27 39 34

   * - Operation
     - Input and processing
     - Result
   * - ``upload(particles, execution)``
     - Copies a host vector into retained host staging, submits a deep copy
       to execution-space AoS storage, then unpacks records into the current
       field arrays.
     - Returns ``void``; sets the active size. Work is ordered on the bound
       execution instance and can still be pending when the call returns.
   * - ``upload`` with a resident prefix
     - Also accepts a ``ParticleState`` view, prefix offset and count.
       The unpack kernel writes resident prefix records before host records.
     - The same current-front representation and stable input order,
       without downloading the resident prefix.
   * - ``download(execution, blocking_wait)``
     - Packs active field-array slots into AoS staging, copies them to host
       staging and waits for completion on the queue's bound execution instance.
     - Returns ``std::vector<ParticleState>`` in current-front order.
       It does not consume or clear the queue.

AoS means *array of structures*: one complete ``ParticleState`` per element.
It is used here for transfer staging, while transport uses the field arrays.
``download`` retains an execution argument for compatibility, but orders its
operations on the instance recorded by the queue, preventing a readback on
an unrelated stream.

``ParticleState`` is trivially copyable, so the queue can copy record bytes
between compatible allocations in one build. Its alignment, padding and
native byte order are not a portable file format. Coordinator frontiers use
explicit versioned fields and environment checks in `StaticFrontier.inl`_,
rather than serializing the struct's memory image.

Resident state progression
--------------------------

The transport loops read ``current()``, write survivors and generated states
to the appropriate output allocations, and then commit the next front.
The particle's history ID follows its state; its storage index and executing
thread can change. A queue is particle storage and bookkeeping, not a fixed
assignment of particles to threads.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Interface
     - Contract
   * - ``current()`` / ``next()``
     - References to the current input and next output field arrays.
   * - ``commitNext(next_size)``
     - Validates the count, fences the bound execution instance, swaps the
       two arrays and records the next active size.
   * - ``commitNextAlreadySynchronized(next_size)``
     - Validates the count and swaps without an additional fence. The caller
       must already have established the required execution ordering.
   * - ``compact(flags)``
     - Scans nonzero flags and scatters retained states in input order.
       Returns the new active count.
   * - ``secondaryOffsets(counts)``
     - Computes exclusive offsets from per-input counts and returns their
       total as ``uint64_t``. These offsets assign storage positions, not
       particle identities.
   * - ``ensureCapacity(requested)``
     - At a synchronized boundary, retains existing storage when sufficient.
       Growth replaces the allocation and resets the active count to zero;
       it does not preserve an active front.

``PendingParticleQueue`` stores numerical records that await a subsequent
batch, including cross-species work. Its prefix download does not consume
records; consumption is explicit. Photons and charged leptons use separate
resident queues. Electrons, positrons and muons share the charged-lepton
path with species-dependent transport logic. See
:doc:`../execution/batches` for dispatch and returned-state policies.

Ownership and synchronization
-----------------------------

The application owns the native environment and event stack. The backend
owns its resident queues and workspaces. Kokkos views manage the underlying
array allocations; raw views borrow them. Host staging remains alive while
asynchronous transfers use it.

Queue reuse requires completion or correct ordering of previous work.
``markExecutionSynchronized()`` only updates queue bookkeeping; it is not a
fence. A caller must establish completion before marking it. Reallocation,
clear and a new upload must respect the queue's synchronization contract.
Retaining allocations between events does not retain a live particle from a
completed event; event state is reset through the backend lifecycle described
in :doc:`../execution/coordinator`.

Extending particle state
------------------------

Adding a field affects more than the record definition. Review the following
interfaces together:

1. Define its physical meaning, unit, frame, default value and lifetime.
   Decide whether it belongs to each particle or to shared physics/environment
   data.
2. Update ``ParticleState``, both SoA representations, constructors,
   ``load``/``store``, raw-view construction and memory accounting. Preserve
   explicit initialization of active slots.
3. Update the value converters and production router transfer where the field
   belongs to their contracts. Handle native stack insertion, surviving
   particles, secondary creation and CPU return paths consistently.
4. Review the static-frontier field format and version policy. The current
   format does not serialize every struct member; a new required field needs
   an explicit import/export decision.
5. Extend round-trip, metadata, coordinate and queue tests. Check each
   supported execution space, then perform physics regression tests if the
   field affects transport or sampling.

The existing tests are mapped in :ref:`particles-validation`. A successful
record round trip establishes a data contract; it does not establish new
particle-species or physical-model support.

.. _WavefrontQueue.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/WavefrontQueue.inl
.. _StaticFrontier.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/scheduling/StaticFrontier.inl
