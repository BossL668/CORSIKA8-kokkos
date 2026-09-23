# Host-only safeguards are independent of the selected accelerator package.
add_executable(testGrazingLayerExit testGrazingLayerExit.cpp)
target_include_directories(testGrazingLayerExit PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})
add_dependencies(testGrazingLayerExit GenParticlesHeaders)
target_compile_features(testGrazingLayerExit PRIVATE cxx_std_17)
target_link_libraries(testGrazingLayerExit PRIVATE CORSIKA8)
if(CORSIKA_ENABLE_KOKKOS AND CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
  set_source_files_properties(testGrazingLayerExit.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(testGrazingLayerExit PROPERTIES CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
endif()
add_test(NAME testGrazingLayerExit COMMAND testGrazingLayerExit)

foreach (cooperative_test IN ITEMS testCooperativeScheduling testCooperativeScheduleJournal testIndependentEndpointDriver testIndependentSubshowerPump testAdaptiveSubshowerControl)
  add_executable (${cooperative_test} ${cooperative_test}.cpp)
  target_include_directories (${cooperative_test} PRIVATE ${CMAKE_SOURCE_DIR})
  target_compile_features (${cooperative_test} PRIVATE cxx_std_17)
  add_test (NAME ${cooperative_test} COMMAND ${cooperative_test})
endforeach ()
find_package(Threads REQUIRED)
target_link_libraries(testIndependentEndpointDriver PRIVATE Threads::Threads)
target_link_libraries(testIndependentSubshowerPump PRIVATE Threads::Threads)

# Selection is a host-only API: test every compiled configuration without
# pretending that this proves HIP/SYCL device compilation or runtime support.
foreach (selection_backend IN ITEMS CUDA_OPENMP CUDA OPENMP HIP SYCL)
  set(selection_target testKokkosExecutionSelection_${selection_backend})
  add_executable(${selection_target} testKokkosExecutionSelection.cpp
    ${CMAKE_SOURCE_DIR}/src/accelerator/em/kokkos/KokkosBackendSelection.cpp)
  target_include_directories(${selection_target} PRIVATE ${CMAKE_SOURCE_DIR})
  target_compile_features(${selection_target} PRIVATE cxx_std_17)
  target_compile_definitions(${selection_target} PRIVATE
    CORSIKA8_KOKKOS_BACKEND_${selection_backend}=1)
  add_test(NAME ${selection_target} COMMAND ${selection_target})
endforeach ()

if (CORSIKA_ENABLE_KOKKOS)
  add_executable(testProfileCrossingModes EXCLUDE_FROM_ALL testProfileCrossingModes.cpp)
  target_link_libraries(testProfileCrossingModes PRIVATE CORSIKA8KokkosEm)
  add_dependencies(testProfileCrossingModes GenParticlesHeaders)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(testProfileCrossingModes.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testProfileCrossingModes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testProfileCrossingModes PROPERTIES CUDA_STANDARD 17
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(testProfileCrossingModes.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testProfileCrossingModes COMMAND testProfileCrossingModes)
  if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
    add_test(NAME testProfileCrossingModesOpenMP COMMAND testProfileCrossingModes openmp)
  endif()
  add_executable(testProposalFallbackVertexGate testProposalFallbackVertexGate.cpp)
  target_link_libraries(testProposalFallbackVertexGate PRIVATE CORSIKA8 PROPOSAL::PROPOSAL)
  add_dependencies(testProposalFallbackVertexGate GenParticlesHeaders)
  add_test(NAME testProposalFallbackVertexGate COMMAND testProposalFallbackVertexGate)

  add_executable(testPhotonFallbackVertex testPhotonFallbackVertex.cpp)
  target_link_libraries(testPhotonFallbackVertex PRIVATE CORSIKA8 Kokkos::kokkos)
  add_dependencies(testPhotonFallbackVertex GenParticlesHeaders)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(testPhotonFallbackVertex.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testPhotonFallbackVertex PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
      target_compile_options(testPhotonFallbackVertex PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    endif()
    target_link_libraries(testPhotonFallbackVertex PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_link_options(testPhotonFallbackVertex PRIVATE "LINKER:--no-as-needed")
    set_target_properties(testPhotonFallbackVertex PROPERTIES
      CUDA_STANDARD 17 CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(testPhotonFallbackVertex.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testPhotonFallbackVertex COMMAND testPhotonFallbackVertex)
  if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
    add_test(NAME testPhotonFallbackVertexOpenMP COMMAND testPhotonFallbackVertex openmp)
  endif()
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^(OPENMP|CUDA_OPENMP)$")
    # Host-only Kokkos instance: no CUDA initialization in this exact-ledger test.
    add_executable(testKokkosHostProfileShards EXCLUDE_FROM_ALL
      testKokkosHostProfileShards.cpp)
    target_link_libraries(testKokkosHostProfileShards PRIVATE CORSIKA8KokkosEm)
    if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
      # The combined Kokkos package requires NVCC to include its public headers;
      # this test still initializes/executes only the OpenMP instance.
      set_source_files_properties(testKokkosHostProfileShards.cpp PROPERTIES LANGUAGE CUDA)
      target_compile_options(testKokkosHostProfileShards PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
      set_target_properties(testKokkosHostProfileShards PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
    endif()
    add_dependencies(testKokkosHostProfileShards GenParticlesHeaders)
    add_test(NAME testKokkosHostProfileShards COMMAND testKokkosHostProfileShards 4)
    set_tests_properties(testKokkosHostProfileShards PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=4;OMP_PROC_BIND=spread;OMP_PLACES=threads")
  endif()
  # Opt-in small runtime/control probe. Not an EM/radio acceptance and not
  # installed with production applications. Use the standalone probe CMake
  # project to exercise this without rebuilding production artifacts.
  if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      find_package(OpenMP REQUIRED COMPONENTS CXX)
      add_executable(testCooperativeDriverAffinity testCooperativeDriverAffinity.cpp)
      target_include_directories(testCooperativeDriverAffinity PRIVATE ${CMAKE_SOURCE_DIR})
      target_compile_features(testCooperativeDriverAffinity PRIVATE cxx_std_17)
      target_link_libraries(testCooperativeDriverAffinity PRIVATE OpenMP::OpenMP_CXX Threads::Threads)
      add_test(NAME testCooperativeDriverAffinity COMMAND testCooperativeDriverAffinity)
      set_tests_properties(testCooperativeDriverAffinity PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=spread;OMP_PLACES=threads")
    endif()
    add_executable(testKokkosCooperativeBackend EXCLUDE_FROM_ALL
      testKokkosCooperativeBackend.cpp CooperativeLeptonFixture.cpp)
    target_link_libraries(testKokkosCooperativeBackend PRIVATE
      CORSIKA8 CORSIKA8KokkosEm)
    add_dependencies(testKokkosCooperativeBackend GenParticlesHeaders)
    option(C8_COOPERATIVE_BACKEND_CUPTI "Instrument isolated cooperative backend test (CUDA 12 only)" OFF)
    if(C8_COOPERATIVE_BACKEND_CUPTI)
      if(NOT CUDAToolkit_VERSION VERSION_LESS "13")
        message(FATAL_ERROR "Cooperative diagnostic needs CUPTI 12 ActivityKernel9")
      endif()
      target_sources(testKokkosCooperativeBackend PRIVATE CooperativeActivityTrace.cpp)
      # Conda's CMake may classify the CUDA include directory as an implicit
      # C++ include although the host compiler does not search it. Compile
      # this CUDA-toolkit-only diagnostic with the CUDA compiler, as in the
      # standalone overlap probes; production has no CUPTI dependency.
      set_source_files_properties(CooperativeActivityTrace.cpp PROPERTIES LANGUAGE CUDA)
      set_target_properties(testKokkosCooperativeBackend PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
      target_link_libraries(testKokkosCooperativeBackend PRIVATE CUDA::cupti)
      target_compile_definitions(testKokkosCooperativeBackend PRIVATE C8_COOPERATIVE_CUPTI=1)
    endif()
    add_executable (testKokkosCooperativeRuntime EXCLUDE_FROM_ALL
      testKokkosCooperativeRuntime.cpp)
    set_source_files_properties (testKokkosCooperativeRuntime.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries (testKokkosCooperativeRuntime PRIVATE CORSIKA8KokkosEm)
    target_compile_options (testKokkosCooperativeRuntime PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_link_options (testKokkosCooperativeRuntime PRIVATE "LINKER:--no-as-needed")
    set_target_properties (testKokkosCooperativeRuntime PROPERTIES
      CUDA_STANDARD 17 CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  endif ()
  add_executable(testKokkosScalarConstants EXCLUDE_FROM_ALL
    testKokkosScalarConstants.cpp ScalarConstantsDriver.cpp)
  add_executable(testKokkosStoppedPositron EXCLUDE_FROM_ALL testKokkosStoppedPositron.cpp)
  target_link_libraries(testKokkosStoppedPositron PRIVATE CORSIKA8KokkosEm)
  add_dependencies(testKokkosStoppedPositron GenParticlesHeaders)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(testKokkosStoppedPositron.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testKokkosStoppedPositron PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosStoppedPositron PROPERTIES CUDA_STANDARD 17
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(testKokkosStoppedPositron.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testKokkosStoppedPositron COMMAND testKokkosStoppedPositron)
  target_link_libraries(testKokkosScalarConstants PRIVATE CORSIKA8 CORSIKA8KokkosEm)
  add_dependencies(testKokkosScalarConstants GenParticlesHeaders)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(ScalarConstantsDriver.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(testKokkosScalarConstants PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(ScalarConstantsDriver.cpp PROPERTIES LANGUAGE HIP)
  endif()
  # Opt-in live CPU final-state audit; an intentional strict failure is data,
  # not a reason to loosen production/test tolerances. Not installed or in ALL.
  add_executable(testKokkosLiveFinalStateOracle EXCLUDE_FROM_ALL
    testKokkosLiveFinalStateOracle.cpp LiveFinalStateDriver.cpp)
  target_link_libraries(testKokkosLiveFinalStateOracle PRIVATE
    CORSIKA8 CORSIKA8KokkosEm CORSIKA8ProposalNative PROPOSAL::PROPOSAL)
  add_dependencies(testKokkosLiveFinalStateOracle GenParticlesHeaders)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(LiveFinalStateDriver.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testKokkosLiveFinalStateOracle PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosLiveFinalStateOracle PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(LiveFinalStateDriver.cpp PROPERTIES LANGUAGE HIP)
  endif()
  # Isolated terrain device test, including the opt-in material step limiter.
  foreach(terrain_test IN ITEMS testKokkosTerrainSession testTerrainCurvedBoundary testKokkosInterfaceQueue testKokkosInterfaceRadio testKokkosInterfaceRadioWaveforms testKokkosInterfaceRadioBvh)
  add_executable(${terrain_test} ${terrain_test}.cpp)
  target_include_directories(${terrain_test} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(${terrain_test} PRIVATE Kokkos::kokkos CORSIKA8)
  if(terrain_test MATCHES "^testKokkosInterfaceRadio")
    target_link_libraries(${terrain_test} PRIVATE CORSIKA8::InterfaceRadio)
    target_compile_definitions(${terrain_test} PRIVATE C8_INTERFACE_EXECUTION_${C8_INTERFACE_EXECUTION_SPACE}=1)
  endif()
  add_dependencies(${terrain_test} GenParticlesHeaders)
  target_compile_features(${terrain_test} PRIVATE cxx_std_17)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(${terrain_test}.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${terrain_test} PRIVATE CUDA::cudart CUDA::cuda_driver)
    # Conan's Kokkos archives are indirect link dependencies. Keep the driver
    # DSO even when CMake places it before those archives under --as-needed.
    if(UNIX AND NOT APPLE)
      target_link_options(${terrain_test} PRIVATE "LINKER:--no-as-needed")
    endif()
    set_target_properties(${terrain_test} PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}" CUDA_STANDARD 17)
    if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
      target_compile_options(${terrain_test} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    endif()
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(${terrain_test}.cpp PROPERTIES LANGUAGE HIP)
  endif()
  if(NOT terrain_test MATCHES "^testKokkosInterfaceRadio(Waveforms|Bvh)$")
  add_test(NAME ${terrain_test} COMMAND ${terrain_test})
  set_tests_properties(${terrain_test} PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  endif()
  endforeach()

  target_sources(testKokkosInterfaceRadioBvh PRIVATE InterfaceRadioBvhMeshLoader.cpp)

  set_tests_properties(testKokkosInterfaceRadio PROPERTIES
    ENVIRONMENT "C8_INTERFACE_TEST_THREADS=${C8_INTERFACE_TEST_THREADS};OMP_NUM_THREADS=${C8_INTERFACE_TEST_THREADS};OMP_PROC_BIND=spread;OMP_PLACES=threads")

  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    target_compile_options(testKokkosInterfaceQueue PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  endif()
  set_tests_properties(testKokkosInterfaceQueue PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=${C8_INTERFACE_TEST_THREADS};OMP_PROC_BIND=spread;OMP_PLACES=threads")
  target_link_libraries(testKokkosInterfaceQueue PRIVATE CORSIKA8InterfaceEm)

  # Opt-in finite convex environment and homogeneous-medium radio checks. These
  # use the same execution-space selection as the regular Kokkos test targets.
  foreach (mountain_test IN ITEMS
      testKokkosConvexEnvironment testKokkosHomogeneousRadio)
    add_executable (${mountain_test} ${mountain_test}.cpp)
    target_link_libraries (${mountain_test} PRIVATE CORSIKA8 CORSIKA8KokkosEm)
    add_dependencies (${mountain_test} GenParticlesHeaders)
    if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
      set_source_files_properties (${mountain_test}.cpp PROPERTIES LANGUAGE CUDA)
      target_compile_options (${mountain_test} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
      set_target_properties (${mountain_test} PROPERTIES
        CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
    elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
      set_source_files_properties (${mountain_test}.cpp PROPERTIES LANGUAGE HIP)
    endif ()
    add_test (NAME ${mountain_test} COMMAND ${mountain_test})
    set_tests_properties (${mountain_test} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
  endforeach ()

  if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
    add_test(NAME Beta5DualBackendRuntime
      COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_dual_backend_runtime.py
        --probe $<TARGET_FILE:kokkos_backend_probe>)
  endif ()
  # These tests compare portable kernels with actual scalar CORSIKA modules,
  # not with a host invocation of the same portable implementation.
  add_executable(testKokkosScalarRadioAlignment testKokkosScalarRadioAlignment.cpp
    ScalarRadioAlignmentDriver.cpp)
  target_link_libraries(testKokkosScalarRadioAlignment PRIVATE CORSIKA8 CORSIKA8KokkosEm)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(ScalarRadioAlignmentDriver.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testKokkosScalarRadioAlignment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosScalarRadioAlignment PROPERTIES CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(ScalarRadioAlignmentDriver.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testKokkosScalarRadioAlignment COMMAND testKokkosScalarRadioAlignment)
  if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
    # DefaultExecutionSpace is CUDA in this build. Explicitly instantiate and
    # run the same oracle for OpenMP as well; a CUDA-only pass is not sufficient.
    add_executable(testKokkosScalarRadioAlignmentOpenMP
      testKokkosScalarRadioAlignment.cpp ScalarRadioAlignmentDriver.cpp)
    target_link_libraries(testKokkosScalarRadioAlignmentOpenMP
      PRIVATE CORSIKA8 CORSIKA8KokkosEm)
    target_compile_definitions(testKokkosScalarRadioAlignmentOpenMP
      PRIVATE C8_TEST_RADIO_OPENMP_EXECUTION=1)
    target_compile_options(testKokkosScalarRadioAlignmentOpenMP
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosScalarRadioAlignmentOpenMP PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
    add_test(NAME testKokkosScalarRadioAlignmentOpenMP
      COMMAND testKokkosScalarRadioAlignmentOpenMP)
  endif()
  add_executable(testKokkosCpuDepositionAlignment testKokkosCpuDepositionAlignment.cpp
    KokkosCpuDepositionAlignmentDriver.cpp)
  target_link_libraries(testKokkosCpuDepositionAlignment PRIVATE CORSIKA8 CORSIKA8KokkosEm)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(KokkosCpuDepositionAlignmentDriver.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testKokkosCpuDepositionAlignment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosCpuDepositionAlignment PROPERTIES CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(KokkosCpuDepositionAlignmentDriver.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testKokkosCpuDepositionAlignment COMMAND testKokkosCpuDepositionAlignment)
  add_executable(testKokkosScalarThinningAlignment testKokkosScalarThinningAlignment.cpp
    KokkosScalarThinningDriver.cpp)
  target_link_libraries(testKokkosScalarThinningAlignment PRIVATE CORSIKA8 CORSIKA8KokkosEm)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(KokkosScalarThinningDriver.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testKokkosScalarThinningAlignment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosScalarThinningAlignment PROPERTIES CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(KokkosScalarThinningDriver.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testKokkosScalarThinningAlignment COMMAND testKokkosScalarThinningAlignment)
  add_executable(testKokkosCpuTransportAlignment
    testKokkosCpuTransportAlignment.cpp KokkosCpuTransportAlignmentDriver.cpp)
  target_link_libraries(testKokkosCpuTransportAlignment PRIVATE CORSIKA8 CORSIKA8KokkosEm
    CORSIKA8ProposalNative PROPOSAL::PROPOSAL spdlog::spdlog CorsikaTestingCommon)
  add_dependencies(testKokkosCpuTransportAlignment GenParticlesHeaders GenMediaProperties)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties(KokkosCpuTransportAlignmentDriver.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(testKokkosCpuTransportAlignment PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(testKokkosCpuTransportAlignment PROPERTIES CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties(KokkosCpuTransportAlignmentDriver.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_test(NAME testKokkosCpuTransportAlignment COMMAND testKokkosCpuTransportAlignment)
  add_test (NAME Beta5UnifiedLauncher
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_beta5_launcher.py)
  add_test (NAME Beta5SourceBoundaries
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/check_beta5_boundaries.py ${CMAKE_SOURCE_DIR})
  add_test (NAME Beta5RejectLegacyPhysicsSource
    COMMAND $<TARGET_FILE:c8_air_shower> --gpu-physics-source c8emrt)
  add_test (NAME Beta5RejectNativeCudaBackend
    COMMAND $<TARGET_FILE:c8_air_shower> --em-backend cuda)
  set_tests_properties (Beta5RejectLegacyPhysicsSource Beta5RejectNativeCudaBackend
    PROPERTIES WILL_FAIL TRUE)

  add_executable (testKokkosTuningCache testKokkosTuningCache.cpp)
  target_link_libraries (
    testKokkosTuningCache PRIVATE CORSIKA8KokkosEm)
  add_test (NAME testKokkosTuningCache COMMAND testKokkosTuningCache)

  add_executable (
    testKokkosPendingParticleQueue testKokkosPendingParticleQueue.cpp)
  target_link_libraries (
    testKokkosPendingParticleQueue PRIVATE CORSIKA8KokkosEm)
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      testKokkosPendingParticleQueue.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      testKokkosPendingParticleQueue PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (
      testKokkosPendingParticleQueue PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      testKokkosPendingParticleQueue.cpp PROPERTIES LANGUAGE HIP)
  endif ()
  add_test (
    NAME testKokkosPendingParticleQueue
    COMMAND testKokkosPendingParticleQueue)

  add_executable (
    testKokkosWavefrontBucketing testKokkosWavefrontBucketing.cpp)
  target_link_libraries (
    testKokkosWavefrontBucketing PRIVATE CORSIKA8KokkosEm)
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      testKokkosWavefrontBucketing.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      testKokkosWavefrontBucketing PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (
      testKokkosWavefrontBucketing PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      testKokkosWavefrontBucketing.cpp PROPERTIES LANGUAGE HIP)
  endif ()
  add_test (
    NAME testKokkosWavefrontBucketing
    COMMAND testKokkosWavefrontBucketing)

  add_executable (
    testKokkosResidentMemoryProjection
    testKokkosResidentMemoryProjection.cpp)
  target_link_libraries (
    testKokkosResidentMemoryProjection PRIVATE CORSIKA8KokkosEm)
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      testKokkosResidentMemoryProjection.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      testKokkosResidentMemoryProjection PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (
      testKokkosResidentMemoryProjection PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      testKokkosResidentMemoryProjection.cpp PROPERTIES LANGUAGE HIP)
  endif ()
  add_test (
    NAME testKokkosResidentMemoryProjection
    COMMAND testKokkosResidentMemoryProjection)

  add_executable (
    testKokkosCompositeExclusiveScan
    testKokkosCompositeExclusiveScan.cpp)
  target_link_libraries (
    testKokkosCompositeExclusiveScan PRIVATE CORSIKA8KokkosEm)
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      testKokkosCompositeExclusiveScan.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      testKokkosCompositeExclusiveScan PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (
      testKokkosCompositeExclusiveScan PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      testKokkosCompositeExclusiveScan.cpp PROPERTIES LANGUAGE HIP)
  endif ()
  add_test (
    NAME testKokkosCompositeExclusiveScan
    COMMAND testKokkosCompositeExclusiveScan)

  add_executable (
    testKokkosFusedProfileStatistics
    testKokkosFusedProfileStatistics.cpp)
  target_link_libraries (
    testKokkosFusedProfileStatistics PRIVATE CORSIKA8KokkosEm)
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      testKokkosFusedProfileStatistics.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      testKokkosFusedProfileStatistics PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (
      testKokkosFusedProfileStatistics PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      testKokkosFusedProfileStatistics.cpp PROPERTIES LANGUAGE HIP)
  endif ()
  add_test (
    NAME testKokkosFusedProfileStatistics
    COMMAND testKokkosFusedProfileStatistics)

  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    add_executable (
      benchmarkKokkosWavefrontBucketing
      benchmarkKokkosWavefrontBucketing.cpp)
    target_link_libraries (
      benchmarkKokkosWavefrontBucketing PRIVATE CORSIKA8KokkosEm)
    set_source_files_properties (
      benchmarkKokkosWavefrontBucketing.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      benchmarkKokkosWavefrontBucketing PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (
      benchmarkKokkosWavefrontBucketing PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  endif ()

  add_executable (
    testKokkosProposalNativeTable
    testKokkosProposalNativeTable.cpp
    KokkosProposalNativeTableTestDriver.cpp
    )
  target_link_libraries (
    testKokkosProposalNativeTable
    PRIVATE
    CORSIKA8KokkosEm
    CORSIKA8ProposalNative
    PROPOSAL::PROPOSAL
    spdlog::spdlog
    )
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      KokkosProposalNativeTableTestDriver.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options (
      testKokkosProposalNativeTable PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
      )
    set_target_properties (
      testKokkosProposalNativeTable PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}"
      )
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      KokkosProposalNativeTableTestDriver.cpp PROPERTIES LANGUAGE HIP)
  endif ()
  add_dependencies (
    testKokkosProposalNativeTable
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testKokkosProposalNativeTable
    COMMAND testKokkosProposalNativeTable
    )
endif ()
