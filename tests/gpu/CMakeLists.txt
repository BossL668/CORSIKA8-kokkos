add_executable (
  testGpuEmHost
  testGpuEmHost.cpp
  )
target_include_directories (
  testGpuEmHost
  PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
  )
target_compile_features (testGpuEmHost PRIVATE cxx_std_17)
target_compile_options (testGpuEmHost PRIVATE -g)
target_link_libraries (testGpuEmHost PRIVATE CORSIKA8)
add_dependencies (
  testGpuEmHost
  GenParticlesHeaders
  GenMediaProperties
  )
add_test (NAME testGpuEmHost COMMAND testGpuEmHost)

add_executable (
  testGpuEmRateTable
  testGpuEmRateTable.cpp
  )
target_link_libraries (testGpuEmRateTable PRIVATE CORSIKA8GpuEmTables)
target_compile_features (testGpuEmRateTable PRIVATE cxx_std_17)
target_compile_options (testGpuEmRateTable PRIVATE -g)
add_test (NAME testGpuEmRateTable COMMAND testGpuEmRateTable)

add_executable (
  testGpuMediumConfig
  testGpuMediumConfig.cpp
  )
target_link_libraries (
  testGpuMediumConfig
  PRIVATE
  CORSIKA8GpuEmTables
  )
target_compile_features (testGpuMediumConfig PRIVATE cxx_std_17)
target_compile_options (testGpuMediumConfig PRIVATE -g)
target_compile_definitions (
  testGpuMediumConfig
  PRIVATE
  C8_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  )
add_test (NAME testGpuMediumConfig COMMAND testGpuMediumConfig)
add_test (
  NAME testGpuEmTablePrepareDryRun
  COMMAND
  $<TARGET_FILE:gpu_em_table_prepare>
  --medium-yaml
  ${CMAKE_SOURCE_DIR}/configs/media/air_dry_1_atm.yaml
  --cache-dir
  ${CMAKE_CURRENT_BINARY_DIR}/table_prepare_dry_run_cache
  --primary-energy-eV
  1e19
  --dry-run
  )
set_tests_properties (
  testGpuEmTablePrepareDryRun
  PROPERTIES
  PASS_REGULAR_EXPRESSION "energy_max_MeV=10500000000000"
  )

add_executable (
  testGpuEmFlatRateTable
  testGpuEmFlatRateTable.cpp
  )
target_link_libraries (
  testGpuEmFlatRateTable
  PRIVATE
  CORSIKA8GpuEmTables
  )
target_include_directories (
  testGpuEmFlatRateTable
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
target_compile_features (testGpuEmFlatRateTable PRIVATE cxx_std_17)
target_compile_options (testGpuEmFlatRateTable PRIVATE -g)
add_test (NAME testGpuEmFlatRateTable COMMAND testGpuEmFlatRateTable)

if (CORSIKA_ENABLE_CUDA)
  add_test (
    NAME testGpuGenericRunnerArchitecture
    COMMAND
    ${CMAKE_COMMAND}
    -DC8_SOURCE_DIR=${CMAKE_SOURCE_DIR}
    -P
    ${CMAKE_CURRENT_SOURCE_DIR}/checkGpuGenericRunnerArchitecture.cmake
    )

  add_executable (
    testGpuEmCppLink
    testGpuEmCppLink.cpp
    )
  target_link_libraries (
    testGpuEmCppLink
    PRIVATE
    CORSIKA8GpuEm
    )
  add_test (NAME testGpuEmCppLink COMMAND testGpuEmCppLink)

  add_executable (
    testGpuHybridRoute
    testGpuHybridRoute.cpp
    )
  target_link_libraries (
    testGpuHybridRoute
    PRIVATE
    CORSIKA8GpuEm
    Eigen3::Eigen
    spdlog::spdlog
    CorsikaTestingCommon
    )
  target_include_directories (
    testGpuHybridRoute
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tests/framework
    )
  add_dependencies (
    testGpuHybridRoute
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (NAME testGpuHybridRoute COMMAND testGpuHybridRoute)
  set_tests_properties (testGpuHybridRoute PROPERTIES SKIP_RETURN_CODE 77)

  add_executable (
    testGpuProposalCpuFallback
    testGpuProposalCpuFallback.cpp
    )
  target_link_libraries (
    testGpuProposalCpuFallback
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    Eigen3::Eigen
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    CorsikaTestingCommon
    )
  target_include_directories (
    testGpuProposalCpuFallback
    PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/framework
    )
  add_dependencies (
    testGpuProposalCpuFallback
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testGpuProposalCpuFallback
    COMMAND testGpuProposalCpuFallback
    )

  add_executable (
    testGpuEmCuda
    testGpuEmCuda.cu
    )
  target_link_libraries (
    testGpuEmCuda
    PRIVATE
    CORSIKA8GpuEm
    )
  set_target_properties (
    testGpuEmCuda
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (NAME testGpuEmCuda COMMAND testGpuEmCuda)
  set_tests_properties (testGpuEmCuda PROPERTIES SKIP_RETURN_CODE 77)

  add_executable (
    testGpuEmRateTableCuda
    testGpuEmRateTableCuda.cu
    )
  target_link_libraries (
    testGpuEmRateTableCuda
    PRIVATE
    CORSIKA8GpuEm
    )
  target_include_directories (
    testGpuEmRateTableCuda
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  set_target_properties (
    testGpuEmRateTableCuda
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (NAME testGpuEmRateTableCuda COMMAND testGpuEmRateTableCuda)
  set_tests_properties (
    testGpuEmRateTableCuda
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuProposalNativeTable
    testGpuProposalNativeTable.cpp
    )
  target_link_libraries (
    testGpuProposalNativeTable
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuProposalNativeTable
    PRIVATE
    ${PROPOSAL_INCLUDE_DIRS}
    ${CubicInterpolation_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    )
  add_dependencies (
    testGpuProposalNativeTable
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testGpuProposalNativeTable
    COMMAND testGpuProposalNativeTable
    )
  set_tests_properties (
    testGpuProposalNativeTable
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuInteractionSelection
    testGpuInteractionSelection.cpp
    )
  target_link_libraries (
    testGpuInteractionSelection
    PRIVATE
    CORSIKA8GpuEm
    )
  target_include_directories (
    testGpuInteractionSelection
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_test (
    NAME testGpuInteractionSelection
    COMMAND testGpuInteractionSelection
    )
  set_tests_properties (
    testGpuInteractionSelection
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuWavefrontBucketing
    testGpuWavefrontBucketing.cpp
    )
  target_link_libraries (
    testGpuWavefrontBucketing
    PRIVATE
    CORSIKA8GpuEm
    )
  add_test (
    NAME testGpuWavefrontBucketing
    COMMAND testGpuWavefrontBucketing
    )
  set_tests_properties (
    testGpuWavefrontBucketing
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuPhotonPairFinalState
    testGpuPhotonPairFinalState.cpp
    )
  target_link_libraries (
    testGpuPhotonPairFinalState
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuPhotonPairFinalState
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_test (
    NAME testGpuPhotonPairFinalState
    COMMAND testGpuPhotonPairFinalState
    )
  set_tests_properties (
    testGpuPhotonPairFinalState
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuPhotonPairLpm
    testGpuPhotonPairLpm.cu
    )
  target_link_libraries (
    testGpuPhotonPairLpm
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  # The local Conan package exports PROPOSAL include directories only for its
  # RelWithDebInfo configuration.  CUDA tests are built in the parent Release
  # configuration, so make the already-discovered include path explicit.
  target_include_directories (
    testGpuPhotonPairLpm
    PRIVATE
    ${PROPOSAL_INCLUDE_DIRS}
    ${CubicInterpolation_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    )
  set_target_properties (
    testGpuPhotonPairLpm
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (
    NAME testGpuPhotonPairLpm
    COMMAND testGpuPhotonPairLpm
    )
  set_tests_properties (
    testGpuPhotonPairLpm
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuBremsLpm
    testGpuBremsLpm.cu
    )
  target_link_libraries (
    testGpuBremsLpm
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuBremsLpm
    PRIVATE
    ${PROPOSAL_INCLUDE_DIRS}
    ${CubicInterpolation_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    )
  set_target_properties (
    testGpuBremsLpm
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (
    NAME testGpuBremsLpm
    COMMAND testGpuBremsLpm
    )
  set_tests_properties (
    testGpuBremsLpm
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuEpairLpm
    testGpuEpairLpm.cu
    )
  target_link_libraries (
    testGpuEpairLpm
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuEpairLpm
    PRIVATE
    ${PROPOSAL_INCLUDE_DIRS}
    ${CubicInterpolation_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    )
  set_target_properties (
    testGpuEpairLpm
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (
    NAME testGpuEpairLpm
    COMMAND testGpuEpairLpm
    )
  set_tests_properties (
    testGpuEpairLpm
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuEmThinning
    testGpuEmThinning.cu
    )
  target_link_libraries (
    testGpuEmThinning
    PRIVATE
    CORSIKA8GpuEm
    )
  set_target_properties (
    testGpuEmThinning
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (
    NAME testGpuEmThinning
    COMMAND testGpuEmThinning
    )
  set_tests_properties (
    testGpuEmThinning
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuMoliere
    testGpuMoliere.cu
    )
  target_link_libraries (
    testGpuMoliere
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuMoliere
    PRIVATE
    ${PROPOSAL_INCLUDE_DIRS}
    ${CubicInterpolation_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    )
  set_target_properties (
    testGpuMoliere
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (
    NAME testGpuMoliere
    COMMAND testGpuMoliere
    )
  set_tests_properties (
    testGpuMoliere
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuUniformMagneticField
    testGpuUniformMagneticField.cu
    )
  target_link_libraries (
    testGpuUniformMagneticField
    PRIVATE
    CORSIKA8GpuEm
    )
  set_target_properties (
    testGpuUniformMagneticField
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    )
  add_test (
    NAME testGpuUniformMagneticField
    COMMAND testGpuUniformMagneticField
    )
  set_tests_properties (
    testGpuUniformMagneticField
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuBremsFinalState
    testGpuBremsFinalState.cpp
    )
  target_link_libraries (
    testGpuBremsFinalState
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuBremsFinalState
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_test (
    NAME testGpuBremsFinalState
    COMMAND testGpuBremsFinalState
    )
  set_tests_properties (
    testGpuBremsFinalState
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuAnnihilationFinalState
    testGpuAnnihilationFinalState.cpp
    )
  target_link_libraries (
    testGpuAnnihilationFinalState
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuAnnihilationFinalState
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_test (
    NAME testGpuAnnihilationFinalState
    COMMAND testGpuAnnihilationFinalState
    )
  set_tests_properties (
    testGpuAnnihilationFinalState
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuIonizationFinalState
    testGpuIonizationFinalState.cpp
    )
  target_link_libraries (
    testGpuIonizationFinalState
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuIonizationFinalState
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_test (
    NAME testGpuIonizationFinalState
    COMMAND testGpuIonizationFinalState
    )
  set_tests_properties (
    testGpuIonizationFinalState
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuEpairFinalState
    testGpuEpairFinalState.cpp
    )
  target_link_libraries (
    testGpuEpairFinalState
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    )
  target_include_directories (
    testGpuEpairFinalState
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_test (
    NAME testGpuEpairFinalState
    COMMAND testGpuEpairFinalState
    )
  set_tests_properties (
    testGpuEpairFinalState
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuSphericalAtmosphere
    testGpuSphericalAtmosphere.cpp
    )
  target_link_libraries (
    testGpuSphericalAtmosphere
    PRIVATE
    CORSIKA8GpuEm
    Eigen3::Eigen
    spdlog::spdlog
    CorsikaTestingCommon
    )
  add_dependencies (
    testGpuSphericalAtmosphere
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testGpuSphericalAtmosphere
    COMMAND testGpuSphericalAtmosphere
    )
  set_tests_properties (
    testGpuSphericalAtmosphere
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuLeptonTransport
    testGpuLeptonTransport.cpp
    )
  target_link_libraries (
    testGpuLeptonTransport
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    Eigen3::Eigen
    spdlog::spdlog
    CorsikaTestingCommon
    )
  target_include_directories (
    testGpuLeptonTransport
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tests/framework
    ${PROPOSAL_INCLUDE_DIRS}
    ${CubicInterpolation_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    )
  add_dependencies (
    testGpuLeptonTransport
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testGpuLeptonTransport
    COMMAND testGpuLeptonTransport
    )
  set_tests_properties (
    testGpuLeptonTransport
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuPhotonWavefront
    testGpuPhotonWavefront.cpp
    )
  target_link_libraries (
    testGpuPhotonWavefront
    PRIVATE
    CORSIKA8GpuEm
    Eigen3::Eigen
    spdlog::spdlog
    CorsikaTestingCommon
    )
  target_include_directories (
    testGpuPhotonWavefront
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
  add_dependencies (
    testGpuPhotonWavefront
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testGpuPhotonWavefront
    COMMAND testGpuPhotonWavefront
    )
  set_tests_properties (
    testGpuPhotonWavefront
    PROPERTIES
    SKIP_RETURN_CODE 77
    )

  add_executable (
    testGpuRadioProjection
    testGpuRadioProjection.cpp
    )
  target_link_libraries (
    testGpuRadioProjection
    PRIVATE
    CORSIKA8GpuEm
    Boost::filesystem
    Eigen3::Eigen
    Parquet::parquet_static
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    )
  add_dependencies (
    testGpuRadioProjection
    GenParticlesHeaders
    GenMediaProperties
    )
  add_test (
    NAME testGpuRadioProjection
    COMMAND testGpuRadioProjection
    )
  set_tests_properties (
    testGpuRadioProjection
    PROPERTIES
    SKIP_RETURN_CODE 77
    )
endif ()
