.. _backend-device-code:

Device-code contracts and portability
=====================================

Backend portability depends on callable operations, accessible storage and
correct execution ordering. Selecting a Kokkos execution space supplies a
dispatch mechanism; it does not make an arbitrary native C8 object graph
usable by a device. The contracts below describe the implementation at
`revision d5297c43 <https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397>`_.

Callable functions and kernel captures
--------------------------------------

The numerical transport functions use ``KOKKOS_INLINE_FUNCTION`` and kernels
use ``KOKKOS_LAMBDA``. The annotations make the functions available to the
selected compilation targets; every operation reached from a kernel must
also be callable there. The lambda macro captures values, so copying a
pointer captures its address rather than transferring its allocation. See
the `Kokkos function annotations`_ for the underlying rules.

Dependency-free framework algorithms use the smaller C8 annotation in
`HostDevice.hpp`_, which does not require Kokkos headers.
``CORSIKA_HOST_DEVICE`` expands to CUDA/HIP host and device qualifiers under
those compilers; it is empty otherwise. ``inline`` remains explicit at the
function definition. Neither annotation repairs a host-only dependency.
Review the entire call chain when extending a numerical routine, including
math helpers, table access and geometry queries. Keep native stack access,
host containers, logging and exception-based preparation in the host adapter.

For example, `transport_lepton's interface`_ requires accessible table and
environment storage, exclusive output slots, non-aliasing inputs and outputs,
and inspection of its status before consuming a record. Its device body
reports an explicit fallback request; it does not execute the CPU handler.
Adding an annotation to a native particle accessor would not satisfy these
requirements. The representation boundary is described in
:doc:`../particles/representation` and :doc:`../environment/extension`.

Accessible storage and borrowed views
-------------------------------------

An execution space must be able to access each allocation used by its kernel.
This is a property of the execution/memory-space pair, expressible through
``Kokkos::SpaceAccessibility<ExecutionSpace, MemorySpace>::accessible``; a
valid host pointer alone is insufficient. See `Kokkos space accessibility`_.

The batch code allocates views in ``TExecutionSpace::memory_space``.
``ParticleSoA<TMemorySpace>`` owns Kokkos views, while
``ParticleSoARawView`` and ``RawView1D/2D`` borrow their addresses. A raw view
does not retain an owner, transfer data or validate indices. Its allocations
must remain unchanged and alive until all submitted users have completed.
When replacing or growing storage, complete outstanding work and obtain new
borrowed views before launching again. The ownership interface is in
`ParticleSoA.hpp`_ and `RawView.hpp`_.

Do not assume a three-component field is laid out as consecutive triples.
The adapter preserves both strides supplied by the owning Kokkos view:

.. literalinclude:: ../../../corsika/detail/kokkos/runtime/RawView.inl
   :language: cpp
   :lines: 17-21
   :lineno-match:

.. literalinclude:: ../../../corsika/detail/kokkos/runtime/RawView.inl
   :language: cpp
   :lines: 28-32
   :lineno-match:

The data layout can differ between execution spaces. Copying a managed
``Kokkos::View`` handle also shares its allocation rather than copying its
elements; use the appropriate transfer operation when a distinct copy is
required. See the `Kokkos View guide`_ and
:doc:`../particles/storage` for the storage interfaces used here.

Ordering copies, kernels and host access
----------------------------------------

Pass the same execution-space instance through a dependent sequence of
transfers and kernels. ``deep_copy(execution, ...)`` may return before the
copy finishes, but it orders that copy with work submitted to that instance.
Another instance of the same execution-space type does not acquire that
ordering automatically. Host reads require completion of the relevant copy;
host staging buffers must also remain alive while a transfer uses them.
These semantics are specified by `Kokkos deep_copy`_.

The validation batch wrapper demonstrates the upload and launch boundary:

.. literalinclude:: ../../../corsika/detail/kokkos/em/execution/LeptonTransportBatch.inl
   :language: cpp
   :lines: 45-54
   :lineno-match:

After transport and compaction, it copies results and fences the same instance
before reading host mirrors:

.. literalinclude:: ../../../corsika/detail/kokkos/em/execution/LeptonTransportBatch.inl
   :language: cpp
   :lines: 94-104
   :lineno-match:

This is a host-returning validation interface. Resident execution retains
storage across stages, so synchronization belongs at its ownership and
observation boundaries. `WavefrontExecutionGuard.hpp`_ documents that both
the queue and its execution instance must outlive the guard; its destructor
attempts to drain that instance on exceptional exit. Preserve that ordering
when changing workspace lifetimes or error paths. A fence on an unrelated
instance does not establish the required completion.

Shared algorithms and narrow backend adapters
---------------------------------------------

The OpenMP, CUDA, HIP and SYCL factory files instantiate
``BackendInstanceImplementation<TExecutionSpace>``. The shared implementation
owns environment/physics storage and dispatches the same numerical stage
interfaces. For example, `CudaBackendInstance.cpp`_ and
`HipBackendInstance.cpp`_ differ in the execution-space template argument.
Backend changes should remain at this boundary when the stage contract is
unchanged.

Sharing also occurs beneath the native/device boundary. The native and
numerical ``next_environment_boundary`` wrappers include the same
`NextEnvironmentBoundaryBody.inl`_ and supply different access adapters.
This does not mean every native C8 algorithm already shares its implementation
with the numerical backend; the environment and transport chapters identify
the actual shared functions.

There are deliberate backend-specific operations. In
`CompositeExclusiveScan.inl`_, automatic CUDA scans use CUB, while the portable
path uses ``Kokkos::parallel_scan``. The CUDA adapter passes
``execution.cuda_stream()`` to CUB and launches the total calculation on the
same Kokkos instance. Both paths implement the same composite integer-prefix
contract. `ResidentExecutionWait.inl`_ similarly uses CUDA completion events
for cooperative or blocking waits and an execution-instance fence for the
ordinary path. These adapters alter how work is executed or observed, while
retaining the numerical interface.

Keep such adapters small and preserve a portable reference path where one
exists. A new backend needs its own implementation of any required native
operation, or an explicitly supported portable path. A conditional branch
that merely compiles is not evidence that its stream, memory and lifetime
contracts have been validated.

Execution tuning and physical models
------------------------------------

`TuningParameters.hpp`_ records batch size, chunk size, team size, track/observer
tile sizes and queue count. ``make_range_policy`` applies a requested chunk
size only to OpenMP. The bounded policy adds launch bounds only for HIP:

.. literalinclude:: ../../../corsika/detail/kokkos/runtime/RangePolicy.inl
   :language: cpp
   :lines: 29-46
   :lineno-match:

These settings govern scheduling and resource use. Keep physical model
selection, table contents, cuts, units and domain checks in their existing
model/configuration interfaces. A new launch policy must preserve the
algorithm's inputs, outputs, source ordering and failure behavior. Numerical
agreement still needs validation: a different execution order does not
establish bitwise equivalence for floating-point reductions. The radio
chapter describes its specific accumulation and tile policies in
:doc:`../radio/accumulation`.

Test registration and support boundaries
----------------------------------------

Tests that instantiate device kernels or include device-dependent Kokkos
headers must use the selected device compiler. The
`test registration helper`_ assigns CUDA or HIP to the listed source files;
for CUDA it also requests C++17 device compilation, extended lambdas and
``CUDA::cudart``. Register new device-bearing test sources through that
helper. Native model fixtures retain ordinary C++ compilation.

Header compilation checks, runtime lifecycle cases, scan comparisons and
queue/transfer tests are registered in the source tree. Some lifecycle cases
run in separate processes because Kokkos cannot be reinitialized after
finalization. The build/install consumer contracts are additionally gated on
the configured production targets and, for real backend execution, the
native export bridge. See :doc:`../backends` for the source and test map.
Run the relevant registered tests in each intended backend configuration;
an OpenMP run cannot exercise a CUDA-only completion-event branch.

The checked-in `GitLab CI configuration`_ defines ordinary GCC/Clang
configure, build and CTest jobs, but their configure commands do not enable
``CORSIKA_ENABLE_KOKKOS``, whose default is ``OFF``. It contains no explicit
Kokkos backend jobs, and this pinned tree contains no GitHub Actions workflow.
The backend test registrations and consumer contracts therefore describe
available checks, not a passing GPU CI matrix.

Production backend recognition is limited to the device combinations in
`BackendRuntimeSelection.cmake`_. Other Kokkos configurations can retain the
generic runtime through ``AUTO`` or ``NONE`` without acquiring a production
backend. A factory or test registration for a backend is not a claim of
validation on every compiler, accelerator or operating system supported by
Kokkos. Record the concrete toolchain, Kokkos build, device and executed tests
when extending platform coverage.

.. _Kokkos function annotations: https://kokkos.org/kokkos-core-wiki/API/core/macros-special/host_device_macros.html
.. _Kokkos space accessibility: https://kokkos.org/kokkos-core-wiki/API/core/SpaceAccessibility.html
.. _Kokkos View guide: https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html
.. _Kokkos deep_copy: https://kokkos.org/kokkos-core-wiki/API/core/view/deep_copy.html
.. _HostDevice.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/framework/core/HostDevice.hpp#L10-L16
.. _transport_lepton's interface: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/em/algorithms/LeptonTransportStep.hpp#L29-L56
.. _ParticleSoA.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/data/ParticleSoA.hpp
.. _RawView.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/RawView.hpp
.. _WavefrontExecutionGuard.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/scheduling/WavefrontExecutionGuard.hpp
.. _CudaBackendInstance.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/CudaBackendInstance.cpp
.. _HipBackendInstance.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/em/HipBackendInstance.cpp
.. _NextEnvironmentBoundaryBody.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl
.. _CompositeExclusiveScan.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/runtime/CompositeExclusiveScan.inl#L119-L184
.. _ResidentExecutionWait.inl: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/detail/kokkos/runtime/ResidentExecutionWait.inl
.. _TuningParameters.hpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/corsika/kokkos/runtime/TuningParameters.hpp
.. _test registration helper: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/CMakeLists.txt#L14-L28
.. _GitLab CI configuration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/.gitlab-ci.yml
.. _BackendRuntimeSelection.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendRuntimeSelection.cmake
