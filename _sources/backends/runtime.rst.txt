Runtime selection and ownership
===============================

The build determines which execution spaces are available. The application
then chooses its EM and radio implementations, process organization and
visible device before creating a backend. A runtime object owns or borrows
Kokkos initialization; the EM backend owns the execution-space instance and
storage used to advance particles.

Application choices
-------------------

``BackendOptions`` contains application settings without owning Kokkos or
device-runtime objects. These settings describe separate choices:

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Setting or interface
     - Meaning
   * - ``--em-backend proposal``
     - Native C8 transport through ``Cascade::run()``. This is the default.
   * - ``--em-backend kokkos-proposal``
     - The accelerated application path with native PROPOSAL calculator data
       and a resident Kokkos EM backend. Its kernels can run on OpenMP or a
       compiled GPU execution space.
   * - ``--em-backend kokkos-egs4``
     - The separate EGS4 transport path, requiring EGS4 build support. Its
       execution validator accepts OpenMP, CUDA or HIP.
   * - ``--radio-backend cpu|kokkos``
     - Radio projection implementation. The standard air-shower
       ``kokkos-proposal`` path requires ``kokkos`` radio. The library also
       permits an application-supplied host radio adapter through
       ``hostRadioAdapter_``; this is not a command-line capability switch.
   * - ``--kokkos-execution``
     - Selection among execution spaces compiled into the backend. An empty
       setting requests the build default.
   * - ``ExecutionSelection``
     - Direct application execution or a process coordinator. It is obtained
       before model and runtime initialization.

``CascadeDriver::cascade(...)`` returns an adapter whose callback first handles
any CPU-prefix role, then runs native ``Cascade`` for ``proposal`` or the
accelerated adapter otherwise. The driver routes application execution; it
does not change the compiled Kokkos execution space. See
:doc:`../application/event-execution` for the event inputs and their lifetimes.

``validate_backend_cli_options(BackendOptions&, executable)`` returns ``bool``
and reports rejected combinations through the logger. In particular, an
explicit ``--kokkos-execution`` requires a Kokkos EM choice. With
``CORSIKA_ENABLE_KOKKOS=OFF``, requesting ``kokkos-proposal`` or Kokkos radio is
rejected. The shared application interface and native ``proposal`` path
remain available; an unavailable accelerated request does not silently select
native transport. In a Kokkos-enabled build, constructing the application
runtime holder also does not itself start Kokkos for the native path.

Sources: `BackendOptions.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/application/BackendOptions.hpp>`_,
`BackendCli.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/BackendCli.cpp>`_
and `CascadeDriver.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/application/cascade/CascadeDriver.inl>`_.

Resolving a compiled backend
----------------------------

``resolve_execution_backend(std::string const& requested)`` returns the
selected backend name without initializing Kokkos. In an independent build,
an empty request selects the sole compiled backend: ``openmp``, ``cuda``,
``hip`` or ``sycl``. A different nonempty request throws
``std::invalid_argument``.

The experimental ``CUDA_OPENMP`` build accepts ``cuda``, ``openmp``,
``cuda-openmp`` and ``openmp-cuda``; an empty request selects ``cuda``.
``EmBackend`` sends the latter two names to the cooperative backend factory.
Constructing ``BackendRuntime`` directly with either cooperative name is
rejected: that runtime represents one endpoint, while the factory assembles
the cooperating endpoints.

The dual build requires a working NVIDIA device during Kokkos initialization,
including when its EM backend selects OpenMP. It does not provide automatic
CPU fallback after device failure. An independent OpenMP build supplies the
CPU-only backend. Native ``proposal`` execution is a separate application
choice and does not imply use of Kokkos OpenMP.

``Kokkos::DefaultExecutionSpace`` is a type fixed by the Kokkos build. It must
not be used as a synonym for the runtime-selected EM backend in the dual
build: ``EmBackend`` explicitly instantiates the selected implementation,
and ``RuntimeInfo::backend_`` reports that selection.

Sources: `BackendSelection.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/BackendSelection.cpp>`_,
`EmBackend.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmBackend.cpp>`_
and `BackendRuntime.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/BackendRuntime.cpp>`_.

.. _backend-device-visibility:

Device selection and visibility
-------------------------------

``parse_devices(argc, argv)`` returns ``DeviceSelection``, whose ``ids_`` is an
ordered vector of identifier strings. It parses ``--device`` and the hidden
``--devices`` alias, accepts comma-separated or space-separated values, and
normalizes numeric identifiers. It stops at ``--``; ``--help`` or ``-h`` before
that marker returns an empty selection without querying hardware. Empty,
duplicate, malformed or excessive selections are rejected. At most 255
devices may be listed, and the device-selection options cannot be combined
with the hidden ``--kokkos-device`` option.

Parsing and resolving identifiers are separate operations:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Backend
     - Resolution
     - Visibility in the application or worker
   * - CUDA
     - ``resolve_gpu_ids(requested, listing)`` maps indices or complete UUID
       aliases from an ``nvidia-smi`` index/UUID listing to UUID strings. It
       rejects unknown identifiers and aliases naming the same GPU.
     - ``CUDA_VISIBLE_DEVICES`` is set to the resolved UUID.
   * - HIP
     - ``resolve_hip_indices(requested, listing)`` validates nonnegative
       ordinal strings against the index/PCI-address listing from
       ``c8_hip_devices``. It returns the requested order unchanged and
       rejects unknown or duplicate indices. UUID input is not accepted.
     - ``HIP_VISIBLE_DEVICES`` is set to the validated index.

``select_single_gpu(DeviceSelection const&)`` requires exactly one identifier
and returns ``void``. It runs the query in a child process, validates the
result, then updates the visibility environment. Query failure, timeout or
failure to set the environment is an error. The selecting parent does not
initialize a device runtime as part of this query.

``prepare_device_cli(app, options, selection)`` returns ``true`` for an empty
selection. For a single device, it checks the compiled backend and compatible
application options, selects visibility and sets ``options.kokkosDevice_`` to
zero. Where the user supplied no explicit choices, it selects
``kokkos-proposal``, Kokkos radio unless a host adapter is supplied, and the
compiled GPU execution name. It catches selection errors, logs them and
returns ``false``. Multiple devices must have been dispatched to the
coordinator before this direct-path helper is called.

The selected device is therefore addressed as ordinal ``0`` inside its
restricted process. ``RuntimeConfig::device_`` and the legacy
``--kokkos-device`` refer to this process-visible ordinal, not the
identifier originally supplied to ``--device``. Device selection neither recompiles
kernels nor changes their architecture target.

Sources: `DeviceSelection.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/DeviceSelection.cpp>`_
and `DeviceCliOptions.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/application/DeviceCliOptions.cpp>`_.

.. _generic-kokkos-runtime:

Generic runtime lifetime
------------------------

``Runtime(int numThreads = 0, int deviceId = -1)`` manages or borrows the one
Kokkos runtime in a process. The first instance initializes Kokkos if needed;
subsequent compatible instances share its lifetime. If Kokkos was initialized
externally, these instances borrow it and never finalize that initialization.
The last reference to an initialization owned by ``Runtime`` fences and
finalizes Kokkos.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Interface or input
     - Contract
   * - ``numThreads``
     - Host parallel-backend thread count. Zero keeps the Kokkos default or
       accepts the existing count. Negative values are invalid; a Serial host
       backend rejects values greater than one.
   * - ``deviceId``
     - Visible GPU ordinal. Minus one keeps the default or accepts the
       existing device. Values below minus one and unavailable explicit
       devices are invalid. A CPU-only build rejects explicit GPU ordinals.
   * - ``getNumThreads()`` / ``getDeviceId()``
     - Return the actual host concurrency and visible device ordinal captured
       from Kokkos; the device value is minus one for a CPU-only build.
   * - ``getExecutionSpaceName()``
     - Returns the build-selected ``Kokkos::DefaultExecutionSpace`` name.
   * - ``ownsInitialization()``
     - Returns whether the shared lifetime owns initialization. All instances
       sharing an owned lifetime report ``true``.

Explicit settings must match an existing runtime. An incompatible request
throws ``std::invalid_argument``. Acquisition from another host thread or
after Kokkos finalization throws ``std::logic_error``. Initialization cannot
be restarted after finalization.

``Runtime`` is neither copyable nor movable. Construct and destroy its
instances on the coordinating host thread. An external owner must belong to
that thread and outlive every borrower. Destroy Kokkos Views and
execution-space objects before releasing the last owning instance.
Wrong-thread destruction violates the contract and terminates the process.

Sources: `Runtime.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/Runtime.hpp>`_
and `Runtime.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/runtime/Runtime.cpp>`_.

.. _backend-runtime-owner:

Backend runtime lifetime
------------------------

``BackendRuntime(RuntimeConfig const&)`` owns the runtime used by an EM
backend endpoint. Ordinary construction requires sole ownership of Kokkos
initialization: an already initialized runtime without an explicit cooperative
lease causes ``std::logic_error``. It does not adopt an arbitrary generic
``Runtime`` or external ``Kokkos::ScopeGuard`` as its owner.

The reverse order is supported: a generic ``Runtime`` can borrow an existing
``BackendRuntime``. That borrower does not extend the backend owner's
lifetime and must be destroyed before the backend owner finalizes Kokkos.
``BackendRuntime`` is noncopyable and movable; its owned implementation and
runtime lease move with it.

For the experimental dual build, a configuration with
``cooperativeOwner_ = true`` selects CUDA and may call
``shareCooperativeLifetime()`` to obtain a ``std::shared_ptr<RuntimeLease>``.
Another endpoint receives that lease through ``RuntimeConfig::runtimeLease_``.
The borrower must use the same coordinating thread and device, a compatible
thread count and a live cooperative lease. Owning and borrowing simultaneously
is invalid. Other builds reject cooperative ownership or borrowing, and an
ordinary owner cannot lend a cooperative lifetime.

The last lease releases the owned initialization. All endpoints, their work,
Views and execution-space objects must finish and be destroyed before that
final lease, on the coordinating thread. This lifetime mechanism alone does
not schedule a cooperative shower. The process coordinator described below
also has a separate role from this in-process CUDA/OpenMP mechanism.

``getInfo()`` returns ``RuntimeInfo const&`` owned by the runtime. It reports
the selected backend, Kokkos and compiler versions, project revision, device
description, concurrency and memory information. ``device_`` retains the
configured ordinal; it is not a physical-device identifier. The reference
remains valid while its owning implementation remains alive.

Sources: `BackendRuntime.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/BackendRuntime.hpp>`_,
`RuntimeConfig.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/RuntimeConfig.hpp>`_
and `RuntimeInfo.hpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/RuntimeInfo.hpp>`_.

Session and execution-space ownership
-------------------------------------

For ``kokkos-proposal``, ``CascadeRunSession`` copies application settings into
the configuration retained by ``EmRunSession``:

.. literalinclude:: ../../../src/kokkos/application/cascade/CascadeRunSession.cpp
   :language: cpp
   :start-at: corsika::kokkos::RuntimeConfig runtimeConfig{};
   :end-at: std::make_unique<corsika::kokkos::EmRunSession>(runtimeConfig, auxiliaryCache);
   :lineno-match:

This constructs a session holder. Its first ``beginProposalNative(...)``
creates ``EmBackend``, selects a backend instance and initializes its storage.
Later compatible events reuse that backend. The application session and
backend storage lifetimes can therefore span several events; they are
not created once per particle or per host thread.

``BackendInstanceImplementation<TExecutionSpace>::Impl`` owns a
``BackendRuntime``, one ``TExecutionSpace execution_`` and the tables, queues
and accumulators used with that instance. Its member order places the runtime
before the execution-space and storage members, so those members are destroyed
before the runtime. Resident transport and accumulator calls receive the backend's
``execution_`` instance; storage upload helpers also manage their own
copy and synchronization paths. The session API and event reuse rules are described
in :ref:`application-backend-start`.

Sources: `EmRunSession.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/EmRunSession.cpp>`_
and `BackendInstanceImplementation.inl
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl>`_.

Host threads and memory configuration
-------------------------------------

``--kokkos-num-threads`` becomes ``RuntimeConfig::threads_``. A positive
value requests Kokkos host threads; zero requests the backend's default
thread policy. Independent
GPU builds use a Serial host, and ordinary GPU execution rejects more than
one host thread. The dual build's ordinary CUDA mode explicitly initializes
OpenMP with one thread. Its cooperative owner may use several OpenMP threads.

This count is distinct from GPU kernel team size, execution-space
``concurrency_`` and the number of CPU-prefix processes. ``RuntimeInfo``
records ``hostThreads_`` separately: normally one for a GPU endpoint and the
OpenMP concurrency for CPU execution; cooperative endpoints report the
shared host-thread count.

``--gpu-memory-fraction`` becomes ``BackendConfig::memoryFraction_``, not a
Kokkos initialization setting. The direct CLI defaults to ``0.70`` and accepts
``[0.01, 1]``; the backend interface accepts ``(0, 1]``. For CUDA and HIP, the
backend computes its byte budget from the free-memory snapshot recorded by
``BackendRuntime`` at initialization. For SYCL, the recorded basis is total
global device memory because the implementation has no portable free-memory
query. This is a budgeting input, not a reservation or a guarantee that a
later allocation succeeds.

The resident implementation checks projected storage against its budget and
derives automatic GPU capacity when ``residentBatchLimit_ == 0``. OpenMP
uses its separate capacity policy; this fraction is not a measured host-RAM
budget. Tuning-cache path and requirement settings also pass through
``RuntimeConfig``, but cache loading belongs to backend initialization, after
runtime and device identity are available.

Sources: `BackendRuntime.cpp`_ and `BackendInstanceImplementation.inl`_.

Process coordinator support
---------------------------

The current GPU coordinator supports the compiled CUDA or HIP path. It
selects ``gpu_execution()`` and ``gpu_visibility_variable()`` accordingly.
It requires Kokkos radio and ``kokkos-proposal`` or enabled ``kokkos-egs4``
transport; it does not dispatch OpenMP or SYCL worker backends. Multiple GPUs,
or one GPU with CPU-prefix workers, enter this path when coordinator support
is compiled. The EGS4 route uses a serial prefix with multiple GPUs;
CPU-prefix workers and persistent GPU-worker reuse are limited to
``kokkos-proposal``. EGS4 also rejects an explicit
``--multigpu-frontier-energy``, because it exports unstarted roots.

Each GPU worker is a separate process with a restricted visibility mask and
``--kokkos-device 0``. The coordinator supplies
``--kokkos-num-threads 1`` and limits the worker's common host-thread
environment settings. CPU-prefix processes have independent process counts.
The coordinator's per-GPU memory fraction defaults to ``0.5`` and accepts
``(0, 0.9]``, a different policy from the direct CLI.

The full worker lifecycle, output merging and supported event constraints
belong to :doc:`../execution/coordinator`. The device-parser tests cover
identifier handling with supplied listings; the CUDA query tests also use a
synthetic query process. These checks do not establish full-shower correctness
or hardware coverage for every compiled backend.

Source: `AirShowerCoordinator.cpp
<https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/scheduling/AirShowerCoordinator.cpp>`_.
