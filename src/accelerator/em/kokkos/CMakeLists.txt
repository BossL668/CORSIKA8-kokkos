add_library (
  CORSIKA8KokkosEm
  STATIC
  KokkosEmBackend.cpp
  KokkosEmRunSession.cpp
  KokkosRuntime.cpp
  )
add_library (CORSIKA8::KokkosEm ALIAS CORSIKA8KokkosEm)

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
  CORSIKA8KokkosEm PUBLIC Kokkos::kokkos CORSIKA8GpuEmTables
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
  CORSIKA8_KOKKOS_BACKEND_${CORSIKA_KOKKOS_BACKEND}=1
  )

if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA")
  set_source_files_properties (
    KokkosEmBackend.cpp
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
  set_source_files_properties (
    KokkosEmBackend.cpp
    KokkosRuntime.cpp
    PROPERTIES LANGUAGE HIP
    )
endif ()
