add_library (
  CORSIKA8GpuEmTables
  STATIC
  FlatRateTable.cpp
  MediumConfig.cpp
  ProposalMedium.cpp
  ../ProposalNativeTable.cpp
  ../ProposalNativeAux.cpp
  ProposalNativeRequirements.cpp
  RateTable.cpp
  Sha256.cpp
  )

add_library (CORSIKA8::GpuEmTables ALIAS CORSIKA8GpuEmTables)

# Native-PROPOSAL export and hashing are host-only shared infrastructure used
# by both native CUDA and Kokkos execution backends.
add_dependencies (CORSIKA8GpuEmTables GenParticlesHeaders)

target_include_directories (
  CORSIKA8GpuEmTables
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

target_compile_features (CORSIKA8GpuEmTables PUBLIC cxx_std_17)
target_link_libraries (
  CORSIKA8GpuEmTables
  PUBLIC
  PROPOSAL::PROPOSAL
  yaml-cpp::yaml-cpp
  PRIVATE
  CubicInterpolation::CubicInterpolation
  nlohmann_json::nlohmann_json
  )
set_target_properties (
  CORSIKA8GpuEmTables
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  )
