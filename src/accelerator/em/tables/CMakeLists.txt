add_library (
  CORSIKA8ProposalNative
  STATIC
  ProposalNativeTable.cpp
  ProposalNativeAux.cpp
  MoliereInterpolation.cpp
  ProposalNativeRequirements.cpp
  Sha256.cpp
  )

add_library (CORSIKA8::ProposalNative ALIAS CORSIKA8ProposalNative)

# Native-PROPOSAL export and hashing are host-only shared infrastructure used
# by both native CUDA and Kokkos execution backends.
add_dependencies (CORSIKA8ProposalNative GenParticlesHeaders)

target_include_directories (
  CORSIKA8ProposalNative
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

target_compile_features (CORSIKA8ProposalNative PUBLIC cxx_std_17)
target_link_libraries (
  CORSIKA8ProposalNative
  PUBLIC
  PROPOSAL::PROPOSAL
  yaml-cpp::yaml-cpp
  PRIVATE
  CubicInterpolation::CubicInterpolation
  nlohmann_json::nlohmann_json
  )
set_target_properties (
  CORSIKA8ProposalNative
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  )
