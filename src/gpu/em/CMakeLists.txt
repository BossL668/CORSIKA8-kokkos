add_library (
  CORSIKA8GpuEm
  STATIC
  CudaEmBackend.cu
  CudaDecisionReplayVerifier.cu
  CudaInteractionSelector.cu
  CudaPhotonTransport.cu
  CudaProfileProjection.cu
  CudaRadioAccumulator.cu
  CudaWavefrontBucketing.cu
  CudaPhotonSelectionTransport.cu
  CudaPhotonPairFinalState.cu
  CudaBremsFinalState.cu
  CudaLeptonTransport.cu
  CudaLeptonVertexSelector.cu
  CudaLeptonSelectionTransport.cu
  MoliereInterpolation.cpp
  CudaRateTable.cu
  )

add_library (CORSIKA8::GpuEm ALIAS CORSIKA8GpuEm)

target_include_directories (
  CORSIKA8GpuEm
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

target_link_libraries (
  CORSIKA8GpuEm
  PUBLIC
  CORSIKA8GpuEmTables
  CUDA::cudart_static
  PRIVATE
  CubicInterpolation::CubicInterpolation
  )

target_compile_features (
  CORSIKA8GpuEm
  PUBLIC
  cxx_std_17
  cuda_std_17
  )

set_target_properties (
  CORSIKA8GpuEm
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
  )

target_compile_definitions (
  CORSIKA8GpuEm
  PUBLIC
  CORSIKA8_WITH_CUDA=1
  )
