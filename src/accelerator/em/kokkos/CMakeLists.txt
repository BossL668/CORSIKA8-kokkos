add_library (
  CORSIKA8KokkosEm
  STATIC
  KokkosEmBackend.cpp
  KokkosBackendSelection.cpp
  KokkosEmRunSession.cpp
  KokkosRuntime.cpp
  KokkosTuningCache.cpp
  )
add_library (CORSIKA8::KokkosEm ALIAS CORSIKA8KokkosEm)
if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
  find_package(Threads REQUIRED)
  target_link_libraries(CORSIKA8KokkosEm PRIVATE Threads::Threads)
  target_sources(CORSIKA8KokkosEm PRIVATE KokkosCooperativeBackend.cpp)
  # CUDA toolkit headers are implicit only for NVCC in the Conda toolchain.
  # The independent driver uses cudaSetDevice; this is dual-mode-only.
  set_source_files_properties(KokkosCooperativeBackend.cpp PROPERTIES LANGUAGE CUDA)
endif ()

target_include_directories (
  CORSIKA8KokkosEm
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )
# The implementation is PIMPL-isolated, but a static archive still needs the
# Kokkos runtime and backend link flags at the final executable link step.
target_link_libraries (
  CORSIKA8KokkosEm PUBLIC Kokkos::kokkos CORSIKA8ProposalNative
  )
target_compile_features (CORSIKA8KokkosEm PUBLIC cxx_std_17)
set_target_properties (
  CORSIKA8KokkosEm PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  )

target_compile_definitions (
  CORSIKA8KokkosEm
  PUBLIC
  CORSIKA8_WITH_KOKKOS=1
  CORSIKA8_PROJECT_REVISION="${CORSIKA8_PROJECT_REVISION}"
  CORSIKA8_KOKKOS_BACKEND_${CORSIKA_KOKKOS_BACKEND}=1
  )

if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA" OR
    CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
  target_sources(CORSIKA8KokkosEm PRIVATE KokkosCudaBackendInstance.cpp)
  set_source_files_properties(KokkosCudaBackendInstance.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties (
    KokkosRuntime.cpp
    PROPERTIES LANGUAGE CUDA
    )
  target_compile_options (
    CORSIKA8KokkosEm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )
  target_link_libraries (
    CORSIKA8KokkosEm PUBLIC CUDA::cudart CUDA::cuda_driver
    )
  set_target_properties (
    CORSIKA8KokkosEm PROPERTIES
    CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}"
    )
elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
  target_sources(CORSIKA8KokkosEm PRIVATE KokkosHipBackendInstance.cpp)
  set_source_files_properties (
    KokkosHipBackendInstance.cpp
    KokkosRuntime.cpp
    PROPERTIES LANGUAGE HIP
    )
endif ()

if (CORSIKA_KOKKOS_BACKEND STREQUAL "OPENMP" OR
    CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
  target_sources(CORSIKA8KokkosEm PRIVATE KokkosOpenMPBackendInstance.cpp)
  if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
    set_source_files_properties(KokkosOpenMPBackendInstance.cpp PROPERTIES LANGUAGE CUDA)
    # Conan's cxxflags apply to CXX sources only. NVCC must also enable its
    # host OpenMP pass when compiling the dual execution-space instances.
    target_compile_options(CORSIKA8KokkosEm PUBLIC
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  endif ()
elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "SYCL")
  target_sources(CORSIKA8KokkosEm PRIVATE KokkosSyclBackendInstance.cpp)
endif ()
