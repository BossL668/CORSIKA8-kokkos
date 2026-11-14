Build configuration and toolchains
==================================

The build selects a Kokkos package, verifies its enabled execution spaces,
and compiles the corresponding CORSIKA implementation. A backend name in
CORSIKA's CMake cache does not enable that backend in an existing Kokkos
library. The package, compiler, architecture and CORSIKA selection must agree.

This page describes the build interfaces at the chapter's pinned source
baseline. For prerequisites and installation commands, see
:doc:`../usage/installation`.

Optional targets and dependencies
---------------------------------

The `top-level CMake configuration`_ defines three options,
all defaulting to ``OFF``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Input
     - Build effect
   * - ``CORSIKA_ENABLE_KOKKOS``
     - Adds ``src/kokkos`` and ``tests/kokkos``, discovers Kokkos and enables
       the required device language before libraries and tests are configured.
   * - ``CORSIKA_ENABLE_PROPOSAL_EXPORT``
     - Requires Kokkos to be enabled. Adds the native export bridge and checks
       the dependency export interface at configuration time.
   * - ``CORSIKA_ENABLE_EGS4``
     - Requires both options above. Adds the native EGS4 implementation;
       selecting a Kokkos execution backend does not enable EGS4 by itself.

The `Kokkos target definitions`_ separate value records, compiled runtime
support and production implementations. The following exported targets
illustrate the dependency boundaries; they are not an exhaustive target list.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Public target
     - Contract
   * - ``CORSIKA8::KokkosData``
     - Interface target providing include paths and C++17. It links neither
       Kokkos nor a physics library, but is created only in a Kokkos-enabled
       CORSIKA build.
   * - ``CORSIKA8::KokkosRuntime``
     - Compiled generic runtime; publicly links ``Kokkos::kokkos`` and
       ``Threads::Threads``.
   * - ``CORSIKA8::NativeTableQueries``
     - Numerical table representation and hashing implementation; publicly
       links ``KokkosData`` and ``Kokkos::kokkos`` without the host export bridge.
   * - ``CORSIKA8::BackendRuntime``
     - Created only when ``CORSIKA_KOKKOS_RESOLVED_BACKEND`` is not ``NONE``.
       Exports the selected production backend definition.
   * - ``CORSIKA8::EmBackend``
     - Created when ``BackendRuntime`` exists. Its compiled sources are
       selected by the resolved backend.
   * - ``CORSIKA8::ProposalBridge`` and ``CORSIKA8::EmRunSession``
     - ``ProposalBridge`` requires the export option. ``EmRunSession`` also
       requires ``EmBackend`` and links the backend to native table preparation.

An enabled build also requires CubicInterpolation for
``CORSIKA8::MoliereInterpolation`` and OpenSSL plus ``nlohmann_json`` for
application support. These dependencies are present even when the native
PROPOSAL export bridge is disabled. Conan's ``with_proposal_export`` option
selects the repository's revision-locked PROPOSAL and CubicInterpolation
recipes; native table preparation is described in
:doc:`../physics-data/preparation`.

With ``CORSIKA_ENABLE_KOKKOS=OFF``, CMake omits the Kokkos libraries, test
subtree and installed Kokkos headers. The ordinary application binaries are
still built: `corsika_add_cascade_application(SOURCE_ROOT, NATIVE_TARGET)`_
includes host CLI utilities directly when ``ApplicationSupport`` is absent.
Its Kokkos EM session sources and definitions are added only if
``CORSIKA8::EmRunSession`` exists. The scalar-only build does not export the
``CascadeApplication`` library as a Kokkos interface.

Resolving the production backend
--------------------------------

`BackendRuntimeSelection.cmake`_ takes ``CORSIKA_KOKKOS_BACKEND`` and the
installed Kokkos package's ``Kokkos_DEVICES``. It uppercases, deduplicates and
sorts the device list, then recognizes these exact sets:

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Normalized ``Kokkos_DEVICES``
     - Detected production backend
   * - ``OPENMP`` or ``OPENMP;SERIAL``
     - ``OPENMP``
   * - ``CUDA;SERIAL``
     - ``CUDA``
   * - ``HIP;SERIAL``
     - ``HIP``
   * - ``SERIAL;SYCL``
     - ``SYCL``
   * - ``CUDA;OPENMP`` or ``CUDA;OPENMP;SERIAL``
     - ``CUDA_OPENMP``
   * - Any other set
     - ``NONE``

The default request, ``AUTO``, uses the detected backend. An explicit
``NONE`` omits the production runtime even for a recognized set. Other
requests must match the detected backend or configuration fails; an unknown
backend name is also an error. The assignment to the resolved variable is:

.. literalinclude:: ../../../cmake/BackendRuntimeSelection.cmake
   :language: cmake
   :lines: 32-43
   :lineno-match:

``NONE`` within an enabled build keeps the generic runtime and numerical
targets, with their Kokkos dependency. It differs from disabling Kokkos
entirely. It also does not suppress device-language setup: the top-level
configuration enables CUDA or HIP from ``Kokkos_DEVICES`` before entering
the target definitions.

`BackendRuntimeTargets.cmake`_ publishes
``CORSIKA_KOKKOS_BACKEND_<resolved backend>=1`` and
``CORSIKA_KOKKOS_PROJECT_REVISION`` on ``BackendRuntime``. The latter is a
cache string defaulting to ``unknown``; a build that needs a recorded source
revision must supply it. `BackendApplicationTargets.cmake`_ uses the same
resolved value to add ``OpenMPBackendInstance.cpp``,
``CudaBackendInstance.cpp``, ``HipBackendInstance.cpp`` or
``SyclBackendInstance.cpp``. ``CUDA_OPENMP`` adds both OpenMP and CUDA
implementations and ``CooperativeBackend.cpp``. Runtime selection and
ownership are described in :doc:`runtime`.

Conan, profiles and the build helper
------------------------------------

The configuration passes through three layers:

1. A host profile selects ``&:with_kokkos``, ``&:kokkos_backend`` and
   ``&:kokkos_architecture``. Bundled profiles inherit ``default`` and may
   override compiler settings. For example, the `OpenMP profile`_ selects
   ``openmp`` with architecture ``NONE``; the `combined CUDA/OpenMP profile`_
   selects ``cuda_openmp`` with ``AMPERE80``.
2. The root `Conan recipe`_ forwards backend and architecture to the Kokkos
   dependency in ``Pkg.configure()``. ``Pkg.generate()`` writes the CORSIKA
   options and device architecture variables into ``conan_toolchain.cmake``;
   ``CMakeDeps`` supplies dependency descriptors. When ``with_kokkos`` is
   true, the generated backend name is uppercased for CMake.
3. The `Kokkos Conan recipe`_ translates the dependency options into
   ``Kokkos_ENABLE_*`` and ``Kokkos_ARCH_*`` cache entries, builds a static
   Kokkos library and exposes its own CMake package. Its
   ``cmake_find_mode=none`` setting preserves Kokkos' exported targets and
   ``Kokkos_DEVICES`` metadata instead of reconstructing them through
   ``CMakeDeps``.

The Kokkos recipe enables OpenMP for ``openmp`` and ``cuda_openmp``, CUDA
for ``cuda`` and ``cuda_openmp``, and HIP or SYCL for their respective
choices. Serial is enabled for every choice except ``openmp``. CUDA and HIP
use ``Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE``. CUDA UVM and Kokkos' async
CUDA allocation implementation are disabled. HIP enables
``Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS`` to retain explicit
launch bounds with older HIP compilers.

`conan-install.sh`_ exports the repository's three local recipes and invokes
``conan install --build=missing`` with distinct host and build profiles,
host C++17 and the requested build type. Exporting a recipe alone does not
select it: the root recipe's options determine the dependency graph.

`build_kokkos.sh`_ orchestrates this dependency step, data preparation,
CMake configuration, compilation and installation. It uses Release for
dependencies and CORSIKA, enables Kokkos and the PROPOSAL export bridge,
and passes the generated toolchain to CMake. Its inputs and outputs are:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Input
     - Effect
   * - Backend argument
     - ``openmp``, ``cuda``, ``cuda-openmp``, ``hip`` or ``sycl``; a
       comma-separated list builds each requested installation in sequence.
   * - ``C8_WORKSPACE``
     - Defaults to the source directory's parent. Each backend uses
       ``build/<backend>``, ``build/<backend>/deps`` and ``install/<backend>``
       beneath it.
   * - ``C8_KOKKOS_OPENMP_PROFILE`` / ``C8_KOKKOS_PROFILE``
     - Select the CPU profile / GPU profile by file path or bundled name.
       HIP and SYCL require an explicit GPU profile.
   * - ``C8_BUILD_PROFILE``
     - Selects Conan's build profile, defaulting to ``default``. Existing
       profiles are not overwritten. A missing default is created with
       ``conan profile detect`` only during a normal build; ``--check`` reports
       its absence.
   * - ``C8_BUILD_JOBS`` / ``C8_PYTHON``
     - Positive compile-job count, defaulting to one / build Python,
       defaulting to ``python3`` from the active environment.
   * - ``--check``
     - Checks prerequisites and reports the plan, then exits before data
       preparation, dependency fetching, configuration or compilation.
   * - Additional CMake arguments
     - Appended to the configuration command. They do not rewrite the Conan
       profile or rebuild Kokkos with a different backend.

``openmp,cuda`` produces two separate dependency/build/install trees.
``cuda-openmp`` produces one combined installation. The compatibility entry
`build_kokkos_dual.sh`_ delegates to the latter and maps
``C8_KOKKOS_DUAL_PROFILE`` to ``C8_KOKKOS_PROFILE`` when provided. These are
different build products, so the comma-separated and combined forms are
not interchangeable.

Architecture and compiler contracts
-----------------------------------

The architecture option controls both Kokkos compilation and application
device compilation. The root Conan recipe accepts these explicit mappings:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Backend
     - Kokkos architecture to CMake device target
   * - CUDA / CUDA_OPENMP
     - ``VOLTA70`` → ``70``, ``VOLTA72`` → ``72``, ``TURING75`` → ``75``,
       ``AMPERE80`` → ``80``, ``AMPERE86`` → ``86``, ``ADA89`` → ``89``,
       ``HOPPER90`` → ``90``, ``BLACKWELL120`` → ``120``;
       written to ``CMAKE_CUDA_ARCHITECTURES``.
   * - HIP
     - ``VEGA906`` → ``gfx906``, ``VEGA908`` → ``gfx908``,
       ``VEGA90A`` → ``gfx90a``, ``GFX940`` → ``gfx940``,
       ``GFX942`` → ``gfx942``; written to ``CMAKE_HIP_ARCHITECTURES``,
       ``AMDGPU_TARGETS`` and ``GPU_TARGETS``.

Unrecognized CUDA or HIP architecture values fail toolchain generation.
These mappings describe accepted configuration inputs, not a matrix of
tested hardware.

Without an explicit CUDA profile, the helper queries ``nvidia-smi`` for
compute capability and selects a bundled profile for 7.0, 7.5, 8.0, 8.6,
8.9, 9.0 or 12.0. It rejects mixed or unknown results. ``C8_CUDA_DEVICE``
limits this build-time query to an index or UUID; it does not select the
device used by a later simulation. For directly specified architectures in
the bundled profiles, the helper checks ``nvcc --list-gpu-code`` before
fetching dependencies. Inherited custom profile settings are resolved by
Conan/CMake. In particular, ``VOLTA72`` appears in the recipe mapping but
has no bundled profile or automatic-detection case.

HIP uses two compiler roles. The `DTK HIP profile`_ and `ROCm HIP profile`_
keep ordinary dependencies on ``clang++`` while selecting ``hipcc`` for
the application's C++ compiler. The helper obtains the toolkit and LLVM
paths from ``hipconfig``, sets ``HIPCXX`` to the toolkit's ``clang++`` unless
already supplied, and adds the toolkit CMake paths. It confines HIP target
environment flags to a subshell for that backend, preventing them from
leaking into the next backend in a multi-build request. The profile's
declared compiler version must match the active toolkit. The supplied
`SYCL profile`_ is likewise a compiler/architecture example, not evidence
that its hardware has been validated.

The helper enforces Conan 2 and CMake 3.24 through 3.x. This is stricter
than the top-level file's historical ``cmake_minimum_required(VERSION 3.9)``.
CORSIKA and the Kokkos recipe use C++17, and native model builds still
require Fortran. The Kokkos integration accepts version 4.7 or later within
4.x and explicitly rejects 5.x; the supplied recipe pins the downloaded
archive to ``4.7.03`` with a SHA-256 checksum. Enabling
``CORSIKA_SANITIZERS_ENABLED`` rejects Kokkos packages with CUDA, HIP, SYCL
or OpenMPTarget enabled.

Installed package behavior
--------------------------

Targets are installed through ``CORSIKA8PublicTargets`` and imported with
the ``CORSIKA8::`` namespace. The generated `corsikaConfig.cmake`_ records
the build's enabled features. For a Kokkos-enabled installation it discovers
Kokkos, Threads, CubicInterpolation, OpenSSL and ``nlohmann_json`` before
importing the targets; a CUDA installation also discovers ``CUDAToolkit``.
Thus the dependency-free link interface of ``KokkosData`` does not make the
whole ``find_package(corsika)`` operation dependency-free.

The export-interface flag and header digest are properties of the installed
binary, checked again when a consumer configures. A consumer cannot change
the installed execution backend or enable an omitted bridge through its own
CMake options. Build a corresponding installation and consume its exported
targets with the matching dependency graph.

.. _top-level CMake configuration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/CMakeLists.txt
.. _Kokkos target definitions: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/CMakeLists.txt
.. _corsika_add_cascade_application(SOURCE_ROOT, NATIVE_TARGET): https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/CascadeApplicationTargets.cmake#L10-L62
.. _BackendRuntimeSelection.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendRuntimeSelection.cmake
.. _BackendRuntimeTargets.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendRuntimeTargets.cmake
.. _BackendApplicationTargets.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendApplicationTargets.cmake
.. _OpenMP profile: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/profiles/openmp
.. _combined CUDA/OpenMP profile: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/profiles/cuda-openmp-ampere80
.. _Conan recipe: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/conanfile.py
.. _Kokkos Conan recipe: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/kokkos/conanfile.py
.. _conan-install.sh: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/conan-install.sh
.. _build_kokkos.sh: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tools/build_kokkos.sh
.. _build_kokkos_dual.sh: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tools/build_kokkos_dual.sh
.. _DTK HIP profile: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/profiles/hip-vega906-dtk221001
.. _ROCm HIP profile: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/profiles/hip-vega90a
.. _SYCL profile: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/externals/conan/profiles/sycl-intel-pvc
.. _corsikaConfig.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/corsikaConfig.cmake.in
