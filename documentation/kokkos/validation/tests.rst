Test coverage and execution
===========================

The test suite checks data contracts, numerical operations, transport stages,
runtime ownership and application integration at different levels. A CTest
entry identifies an executable or a selected scenario, not necessarily one
assertion or one Catch2 test case. The configured build determines which
entries exist and which execution space they exercise.

This page describes registrations and assertions in the pinned source baseline.
It is a guide to selecting and interpreting checks, not a record of passing
runs. Numerical comparison rules are covered in :doc:`numerics`; timing and
benchmark evidence are covered in :doc:`performance`.

Where the tests live
--------------------

The `test root`_ adds the ordinary framework, media, stack, module and output
tests, then conditionally adds ``tests/kokkos``. Application-support contracts
are registered separately. The relevant organization is:

.. code-block:: text

   tests/
   ├── kokkos/
   │   ├── CMakeLists.txt        generic Runtime scenarios and subtree registration
   │   ├── data/                record layout, defaults, conversion and transfer
   │   ├── environment/         export, storage, queries and boundary handling
   │   ├── utility/             elementary functions and polynomial roots
   │   ├── em/
   │   │   ├── tables/          native export, lookup and storage contracts
   │   │   ├── models/, moliere/, continuous/, discrete/, stochastic/
   │   │   ├── selection/, transport/, finalstate/
   │   │   ├── execution/      fronts, cascades, statistics, backend and session
   │   │   └── proposal/       native CPU fallback and particle transfer
   │   ├── runtime/            scans, queues, completion waits and BackendRuntime
   │   ├── scheduling/         scalar/hybrid loops and worker protocols
   │   ├── output/             profile projection, storage and accumulation
   │   ├── radio/              projection, accumulation, host staging and firn
   │   ├── application/        scene preparation
   │   ├── diagnostics/        decision replay records
   │   └── egs4/               optional EGS4 tables and transport
   ├── application_support/    CLI, execution selection, bindings and coordinator
   └── modules/, output/, ...  native physics and output integration
   cmake/*Contracts.cmake      shared build-tree/installed-consumer registrations

Files ending in ``Header.cpp`` commonly compile a public header in its own
translation unit. They establish include and link contracts; they are not
separate numerical tests. ``Reference.hpp``, ``reference/`` and ``oracle/``
hold comparison implementations used by the surrounding test. A comparison
against preserved production code establishes compatibility for its inputs;
it does not provide an independent measurement of physical accuracy.

Registration, gates and selectors
---------------------------------

The `top-level CMake configuration`_ calls ``enable_testing()`` and adds
``tests`` unconditionally. There is no ``BUILD_TESTING`` gate around that
subdirectory at this baseline. The `Kokkos test registration`_ uses explicit
``add_test()`` calls, alongside the native ``CORSIKA_ADD_TEST`` helper; it does
not discover each Catch2 case as a separate CTest entry.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Configuration or prerequisite
     - Effect on the selected checks
   * - ``CORSIKA_ENABLE_KOKKOS=ON``
     - Adds the Kokkos subtree. With it disabled, native/application-support
       tests still exist, but the Kokkos subtree does not.
   * - ``CORSIKA_ENABLE_PROPOSAL_EXPORT=ON``
     - Enables real native table fixtures and the export-dependent numerical
       tests. Some existing transport/front/cascade executables gain additional
       native cases through compile definitions, so unchanged test counts do
       not imply unchanged coverage.
   * - Exported ``EmBackend`` and real export support
     - Enable ``production_backend_Stages_<execution>`` and
       ``production_backend_Router_<execution>``. ``EmRunSession`` additionally
       enables the cold/hot, environment and failure scenarios.
   * - ``CORSIKA_ENABLE_EGS4=ON``
     - Requires Kokkos and the PROPOSAL export bridge. The EGS4 test directory
       registers its three executables only when ``CORSIKA8NativeEgs4`` exists.
   * - Unix and FLUKA availability
     - Unix gates the process-pool and hybrid-loop fixtures. The FLUKA worker
       subtree is entered when the cached ``C8_FLUKA_ENABLED`` value is
       nonempty; its tests use the actual worker executables and model setup.
   * - ``CORSIKA_TEST_COORDINATOR_PROCESS=ON``
     - Adds the coordinator's ``process`` comparison. It defaults to ``OFF``;
       its declared prerequisite is ``nvidia-smi`` and NVIDIA devices 0 and 1.
       Ordinary ``io``, ``cli``, ``firn`` and ``prefix`` cases remain registered.
   * - ``CORSIKA_SANITIZERS_ENABLED=ON``
     - Adds address/undefined-behavior sanitizer flags to participating targets.
       ``src/kokkos/CMakeLists.txt`` rejects this configuration with CUDA, HIP,
       SYCL or OpenMPTarget devices; use a separate host-only build.

Sources: `native table registration`_, `execution registration`_,
`backend execution contracts`_, `scheduling registration`_,
`EGS4 registration`_, `coordinator contracts`_ and `Kokkos target definitions`_.

Many numerical tests use ``Kokkos::DefaultExecutionSpace``. Configuring CUDA,
OpenMP or HIP changes the compiled execution space; running the same binary
again does not switch it. The optional ``CORSIKA_DATA_EXPECT_BACKEND`` cache
input adds explicit type assertions to participating tests. Its `data test
registration`_ accepts only an empty string, ``CUDA`` or ``OPENMP``. It neither
builds a backend nor accepts ``HIP`` or ``SYCL`` as a full-tree audit setting.

Backend/session registrations derive their execution arguments from the
compiled capability: ``OPENMP`` registers ``openmp``, and CUDA, HIP and SYCL
each register their corresponding execution. ``CUDA_OPENMP`` registers
``cuda``, ``openmp``, ``cuda-openmp`` and ``openmp-cuda``. These registrations
show intended test paths, not evidence that all paths have run successfully.
In particular, the `radio accumulator registration`_ selects an OpenMP
expectation whenever neither CUDA nor HIP is detected, and its helper asserts
that type. The complete suite therefore must not be assumed to support every
Serial or SYCL configuration unchanged.

CTest labels and names serve different purposes. ``-L kokkos`` selects the
entries carrying that label, including the generic runtime and most numerical
tests. It omits unlabelled ``production_*`` backend/session entries and
application-support contracts. FLUKA worker entries carry ``fluka;worker``.
Use ``ctest -N`` to inspect the actual build before treating a selection as a
complete suite.

Lifecycle tests deliberately use one process per scenario because Kokkos
cannot be reinitialized after finalization. The runtime registration preserves
the Catch2 filter, environment and timeout together:

.. literalinclude:: ../../../tests/kokkos/CMakeLists.txt
   :language: cmake
   :lines: 58-71
   :lineno-match:

The same pattern separates data transfer from device-default construction.
Running ``testKokkosRuntime`` or ``testKokkosDataTransfer`` without a case filter
bypasses this isolation. Use their registered CTest entries. ``RUN_SERIAL``
also prevents these tests from overlapping other CTest work; ``PROCESSORS``
is a scheduler reservation, while ``OMP_NUM_THREADS`` configures host threads.

Representative assertions
-------------------------

This table connects representative CTest entries to their actual assertions.
It is a navigation map, not an exhaustive inventory or a pass/fail report.
Wildcard suffixes denote the separate registrations described above.

.. list-table::
   :header-rows: 1
   :widths: 31 69

   * - CTest entry and source
     - What a successful run establishes for its fixture
   * - ``testKokkosRuntime_*``; `testRuntime.cpp`_
     - Checks owned, shared and borrowed initialization; conflicting and invalid
       options; finalization; coordinator-thread restrictions; exception cleanup;
       and execution of a kernel with checked values and reduction output.
   * - ``testKokkosDataTransfer_transfer`` and
       ``testKokkosDataTransfer_device_defaults``; `testKokkosDataTransfer.cpp`_
     - Copies record fields through the actual execution space, checks transfer
       markers and record contents, constructs defaults inside a kernel, and
       verifies the Runtime lifetime before and after the operation.
   * - ``testKokkosEnvironmentQueryDevice``;
       `testEnvironmentQueryDevice.cpp`_
     - Runs independent candidate/reference queries on exported node-tree data.
       Asserts 160 query cases, a 34-case host/device subset and 15 atmosphere
       integration chains; checks states, values, stored inputs and semantics.
   * - ``testProposalNativeQueriesDevice``;
       `testProposalNativeQueriesDevice.cpp`_
     - Compares the candidate lookup with a preserved same-device implementation;
       requires every query operation, Newton and bisection paths, a nonempty
       host/device subset and unchanged table content hashes. Requires export.
   * - ``testInteractionSelection``; `testInteractionSelection.cpp`_
     - Uses real native tables and checks host/device selection against the
       reference, including continuations, decays, cuts, failure paths,
       bremsstrahlung, ionization, pair creation, Compton scattering and native
       replay for electrons and muons. Requires export.
   * - ``testPhotonFinalState`` and ``testLeptonFinalState``;
       `final-state tests`_
     - Check child counts, identities, directions, thinning/suppression and CPU
       fallback payloads. Export-enabled cases add real native CDF/auxiliary
       inputs. The stopped-positron case checks two equal-energy, opposite
       photon directions; these are local reaction assertions.
   * - ``testLeptonStatistics``; `testMaterialBoundaryStatistics.cpp`_
     - Calls real firn-boundary transport using an exact linear range fixture.
       Checks distance and grammage, preserved identity, deposited energy,
       material-boundary counts, fixed profile equality, zero invalid records,
       unknown-limit rejection and counter-overflow rejection.
   * - ``testRadioAccumulator``; `testRadioMerge.cpp`_
     - Checks exact fixed-integer waveform merging and endpoint identity,
       rejects overflow without partial commit, and exercises an explicit GPU
       stream in the CUDA/HIP branches. Other source files in the executable
       cover accumulation, homogeneous propagation, staging and counters.
   * - ``testFirnTimeDomain``; `testTimeDomainEmission.cpp`_
     - Checks finite exact-cone CoREAS endpoints, ZHS bin-integral conservation,
       window clipping and time moments. This executable also contains
       propagation/emission comparisons; tolerances belong to each assertion.
   * - ``testHadronicProcessPool`` and ``testHybridCascade``;
       `process-pool tests`_ and `hybrid-loop tests`_
     - Check ordering, worker assignment, response contents, rejection and child
       cleanup; compare scalar/routed/deferred coordinator behavior with the
       preserved implementation. Their deterministic protocol workers do not
       replace tests of a physical hadronic generator.
   * - ``testCpuFallbackPhysics``; `testCpuFallbackPhysics.cpp`_
     - Completes real specified PROPOSAL reactions, compares resulting particle
       fields and native random-state replay, and checks generated-secondary
       and completion counters.
   * - ``production_backend_Stages_*`` and
       ``production_session_cold-hot_*``; `BackendStages.cpp`_ and `Session.cpp`_
     - Stage checks partition every input between selection and fallback, verify
       repeatability and native fallback provenance, and clear queues for a new
       shower. Session checks retain backend/table identity and static-upload
       counters across warm showers while repeating named selection results.
   * - ``application_execution_selection``; `ExecutionSelection.cpp`_
     - Checks direct dispatch, malformed-argument failure, and coordinated
       selection when compiled. Its direct runner is a callback fixture; this
       test does not launch a complete air shower.
   * - ``testEgs4Tables``; `testTables.cpp`_
     - Compares embedded and external table sessions for electron, positron and
       photon inputs, checks cut clamping and Sternheimer correction handling,
       and verifies nonempty transport traces. Requires the EGS4 target.

Exact equality in one row does not set a tolerance for another. The numerical
fixtures distinguish state/identity equality, same-device comparison,
host/device tolerance and sampling or physical observables. Follow
:doc:`numerics` before changing a comparison to accommodate a new backend.

Run the configured build
------------------------

Use the compiler environment, dependencies and backend from
:doc:`../usage/installation`. The helper builds and installs but does not run
CTest. From the source directory, an OpenMP build and inspection can be made
explicit as follows; retain any physics options used for the installation:

.. code-block:: bash

   c8_source="$PWD"
   c8_workspace="${C8_WORKSPACE:-$(dirname "$c8_source")}"
   C8_WORKSPACE="$c8_workspace" C8_BUILD_JOBS=1 \
     bash tools/build_kokkos.sh openmp
   c8_build="$c8_workspace/build/openmp"
   ctest --test-dir "$c8_build" -N
   ctest --test-dir "$c8_build" --print-labels
   ctest --test-dir "$c8_build" -N -L kokkos

For CUDA or HIP, use that backend's installation command and corresponding
build directory; the GPU must be accessible during execution. Changing a
CTest selector does not change the physics options or compiler configuration.

Run focused groups during development, then run the complete configured suite
for a release check. The execution commands below use CTest 3.26 or newer:
``--no-tests=error`` makes an empty selection fail instead of appearing
successful. This flag requires a newer CTest than the build helper's 3.24
minimum; see the `CTest command reference`_.

.. code-block:: bash

   ctest --test-dir "$c8_build" -j 1 --output-on-failure --no-tests=error \
     -R '^testKokkosRuntime_'
   ctest --test-dir "$c8_build" -j 1 --output-on-failure --no-tests=error -L kokkos
   ctest --test-dir "$c8_build" -j 1 --output-on-failure --no-tests=error \
     -R '^(production_|application_|native_)'
   ctest --test-dir "$c8_build" -j 1 --output-on-failure --no-tests=error \
     --output-junit "$c8_build/ctest-all.xml" \
     --output-log "$c8_build/ctest-all.log"

The unfiltered command also runs ordinary native tests and any enabled physical
worker tests. It may take substantially longer than a focused selection.
Inspect the listed entries when a gate is disabled: an absent test is not a
passing result. Registered environment settings supply thread counts and
dedicated native/auxiliary/Moliere cache directories where needed. Preserve
those caches when diagnosing a warm-cache failure, and use a fresh dedicated
build/cache location when explicitly measuring cold behavior.

Many Catch2 registrations write their own JUnit reports under
``<build>/test_outputs/junit-<test>.xml``. The CTest-level report above also
records standalone C++ and Python tests. Keep the configure cache, selected
test list, logs, revision, compiler, Kokkos devices, hardware and physics
options with the result. A timing field checked for nonnegativity is not a
performance regression criterion; see :doc:`performance`.

Check an installed consumer
---------------------------

A build-tree test can succeed while an installed target omits an include path,
archive or transitive dependency. The repository supplies shared registration
modules with installed-consumer support, but no standalone installed-consumer
project at this baseline. The consumer must find the installed ``corsika``
package and link its imported targets.

For example, create an external directory
``$c8_workspace/installed-runtime-contract`` and save this as its
``CMakeLists.txt``. ``C8_TEST_SOURCE`` points to the pinned source checkout for
test inputs; candidate headers and libraries come from the installed targets:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.24)
   project(InstalledRuntimeContract LANGUAGES CXX)
   enable_testing()
   find_package(corsika CONFIG REQUIRED)
   set(C8_TEST_SOURCE "" CACHE PATH "CORSIKA source containing test fixtures")
   set(CORSIKA_RUNTIME_CONTRACT_PREFIX installed)
   set(CORSIKA_RUNTIME_CONTRACT_TESTS
       "${C8_TEST_SOURCE}/tests/kokkos/runtime/backend")
   set(CORSIKA_RUNTIME_CONTRACT_REFERENCE
       "${CORSIKA_RUNTIME_CONTRACT_TESTS}/reference")
   include("${C8_TEST_SOURCE}/cmake/BackendRuntimeContracts.cmake")

Using the OpenMP installation and dependency toolchain above:

.. code-block:: bash

   cmake -S "$c8_workspace/installed-runtime-contract" \
     -B "$c8_workspace/installed-runtime-build" \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_TOOLCHAIN_FILE="$c8_build/deps/conan_toolchain.cmake" \
     -DCMAKE_PREFIX_PATH="$c8_workspace/install/openmp" \
     -DC8_TEST_SOURCE="$c8_source"
   cmake --build "$c8_workspace/installed-runtime-build" --parallel 1
   ctest --test-dir "$c8_workspace/installed-runtime-build" -N
   ctest --test-dir "$c8_workspace/installed-runtime-build" \
     -j 1 --output-on-failure --no-tests=error

`BackendRuntimeContracts.cmake`_ checks the published backend definition and
registers cache, ownership, borrowing, unavailable-backend and thread-policy
scenarios. Its cache test includes only the preserved reference directory;
the candidate is linked through ``CORSIKA8::TuningCache``. Without
``CORSIKA8::BackendRuntime``, only the cache contract remains. This example
therefore checks a specific installed interface, not the whole installation.

For a broader consumer, `BackendExecutionContracts.cmake`_ accepts
``BC_INSTALLED=TRUE``, ``BC_NATIVE=CORSIKA8::CORSIKA8`` and
``BC_LIBRARY_PREFIX=CORSIKA8::`` together with its source-fixture paths,
``BC_BACKEND`` and ``BC_REAL_EXPORT``. Obtain these capabilities from the
imported targets and ``CORSIKA8_PROPOSAL_EXPORT_ENABLED``; do not invent a
consumer-selectable production backend. The module rebuilds host table fixtures
against installed ``ProposalBridge``/``MoliereInterpolation`` and links the
actual installed ``EmBackend``/``EmRunSession``. Application support, bindings
and coordinator have analogous shared registrations. These are explicit
consumer checks; running source-tree CTest alone does not exercise their
installed branches.

The installed ``c8_air_shower`` should also be exercised using
:doc:`../usage/first-shower` and the relevant event/coordinator configuration.
Module fixtures, package linkage and complete-shower comparisons answer
different questions. None of the registrations above establishes a universal
cross-backend pass matrix or statistical agreement for full showers.

.. _test root: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/CMakeLists.txt
.. _top-level CMake configuration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/CMakeLists.txt
.. _Kokkos test registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/CMakeLists.txt
.. _native table registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/tables/CMakeLists.txt
.. _execution registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/CMakeLists.txt
.. _backend execution contracts: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendExecutionContracts.cmake
.. _scheduling registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/scheduling/CMakeLists.txt
.. _EGS4 registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/egs4/CMakeLists.txt
.. _coordinator contracts: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/CoordinatorContracts.cmake
.. _Kokkos target definitions: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/src/kokkos/CMakeLists.txt
.. _data test registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/data/CMakeLists.txt
.. _radio accumulator registration: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/CMakeLists.txt
.. _testRuntime.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/runtime/testRuntime.cpp
.. _testKokkosDataTransfer.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/data/testKokkosDataTransfer.cpp
.. _testEnvironmentQueryDevice.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/environment/testEnvironmentQueryDevice.cpp
.. _testProposalNativeQueriesDevice.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/tables/testProposalNativeQueriesDevice.cpp
.. _testInteractionSelection.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/selection/testInteractionSelection.cpp
.. _final-state tests: https://github.com/BossL668/CORSIKA8-kokkos/tree/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/finalstate
.. _testMaterialBoundaryStatistics.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/testMaterialBoundaryStatistics.cpp
.. _testRadioMerge.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/accumulator/testRadioMerge.cpp
.. _testTimeDomainEmission.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/radio/firn/testTimeDomainEmission.cpp
.. _process-pool tests: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/scheduling/testHadronicProcessPool.cpp
.. _hybrid-loop tests: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/scheduling/hybrid/testHybridCascade.cpp
.. _testCpuFallbackPhysics.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/proposal/testCpuFallbackPhysics.cpp
.. _BackendStages.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/backend/BackendStages.cpp
.. _Session.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/em/execution/session/Session.cpp
.. _ExecutionSelection.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/application_support/ExecutionSelection.cpp
.. _testTables.cpp: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/tests/kokkos/egs4/testTables.cpp
.. _BackendRuntimeContracts.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendRuntimeContracts.cmake
.. _BackendExecutionContracts.cmake: https://github.com/BossL668/CORSIKA8-kokkos/blob/d5297c43c041a22b43c65c322d9b51bc7ec8e397/cmake/BackendExecutionContracts.cmake
.. _CTest command reference: https://cmake.org/cmake/help/latest/manual/ctest.1.html
