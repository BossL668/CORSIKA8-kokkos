add_library (
  CORSIKA8GpuEmTables
  STATIC
  FlatRateTable.cpp
  MediumConfig.cpp
  ProposalMedium.cpp
  RateTable.cpp
  Sha256.cpp
  )

add_library (CORSIKA8::GpuEmTables ALIAS CORSIKA8GpuEmTables)

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
  )
set_target_properties (
  CORSIKA8GpuEmTables
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  )
