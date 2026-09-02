add_library (
  CORSIKA8GpuEm
  STATIC
  CudaEmBackend.cu
  CudaEmRunSession.cpp
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
  CudaProposalNativeTable.cu
  )

add_library (CORSIKA8::GpuEm ALIAS CORSIKA8GpuEm)

# ProposalNativeTable.cpp consumes the generated CORSIKA particle/PDG mapping.
# Keep this dependency explicit so building only CORSIKA8GpuEm is reliable.
add_dependencies (CORSIKA8GpuEm GenParticlesHeaders)

target_include_directories (
  CORSIKA8GpuEm
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

# Public C++ headers such as DeviceWorkspace.hpp include cuda_runtime_api.h.
# With some Conda CUDA packages CMake records the toolkit directory as implicit
# for CUDA and removes it from commands compiling ordinary C++ consumers.  Find
# the header without embedding a machine-specific path, then keep it on the C++
# usage interface so applications and tests can include the public GPU headers.
set (_c8_gpu_cuda_runtime_include_dir "")
foreach (_c8_gpu_cuda_include_candidate IN ITEMS
         "${CUDAToolkit_INCLUDE_DIRS}"
         "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
         "${CUDAToolkit_CUPTI_INCLUDE_DIR}")
  if (EXISTS "${_c8_gpu_cuda_include_candidate}/cuda_runtime_api.h")
    set (_c8_gpu_cuda_runtime_include_dir
         "${_c8_gpu_cuda_include_candidate}")
    break ()
  endif ()
endforeach ()
if (NOT _c8_gpu_cuda_runtime_include_dir)
  message (FATAL_ERROR
           "CORSIKA8GpuEm requires a directory containing cuda_runtime_api.h")
endif ()
target_compile_options (
  CORSIKA8GpuEm
  PUBLIC
  "$<$<COMPILE_LANGUAGE:CXX>:-I${_c8_gpu_cuda_runtime_include_dir}>"
  )

target_link_libraries (
  CORSIKA8GpuEm
  PUBLIC
  CORSIKA8GpuEmTables
  CUDA::cudart_static
  PRIVATE
  CubicInterpolation::CubicInterpolation
  PROPOSAL::PROPOSAL
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
