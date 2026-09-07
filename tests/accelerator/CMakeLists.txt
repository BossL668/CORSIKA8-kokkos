if (CORSIKA_ENABLE_KOKKOS)
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
