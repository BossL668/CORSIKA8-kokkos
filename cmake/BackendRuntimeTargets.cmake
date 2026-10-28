# Included from src/kokkos after the generic runtime and numerical targets.
include ("${CMAKE_CURRENT_LIST_DIR}/BackendRuntimeSelection.cmake")

add_library (CORSIKA8TuningCache runtime/TuningCache.cpp)
add_library (CORSIKA8::TuningCache ALIAS CORSIKA8TuningCache)
set_target_properties (CORSIKA8TuningCache PROPERTIES EXPORT_NAME TuningCache)
target_compile_features (CORSIKA8TuningCache PUBLIC cxx_std_17)
target_link_libraries (CORSIKA8TuningCache PUBLIC CORSIKA8NativeTableQueries)
set (CORSIKA_PRODUCTION_RUNTIME_TARGETS CORSIKA8TuningCache)

if (NOT CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "NONE")
  add_library (CORSIKA8BackendRuntime
    runtime/BackendRuntime.cpp runtime/BackendSelection.cpp)
  add_library (CORSIKA8::BackendRuntime ALIAS CORSIKA8BackendRuntime)
  set_target_properties (CORSIKA8BackendRuntime PROPERTIES EXPORT_NAME BackendRuntime)
  target_compile_features (CORSIKA8BackendRuntime PUBLIC cxx_std_17)
  target_link_libraries (CORSIKA8BackendRuntime
    PUBLIC CORSIKA8KokkosData Kokkos::kokkos Threads::Threads)
  set (CORSIKA_KOKKOS_PROJECT_REVISION "unknown" CACHE STRING
    "Source revision of this production runtime build")
  target_compile_definitions (CORSIKA8BackendRuntime PUBLIC
    CORSIKA_KOKKOS_BACKEND_${CORSIKA_KOKKOS_RESOLVED_BACKEND}=1
    CORSIKA_KOKKOS_PROJECT_REVISION="${CORSIKA_KOKKOS_PROJECT_REVISION}")
  if (CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "^CUDA")
    enable_language (CUDA)
    find_package (CUDAToolkit REQUIRED)
    # Host C++ executables also consume these static CUDA implementations.
    # Publish the runtime dependency instead of relying on an nvcc link step.
    target_link_libraries (CORSIKA8BackendRuntime PUBLIC CUDA::cudart)
    target_link_libraries (CORSIKA8KokkosRuntime PUBLIC CUDA::cudart)
    set_target_properties (CORSIKA8BackendRuntime PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    set_target_properties (CORSIKA8KokkosRuntime PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    set_source_files_properties (runtime/BackendRuntime.cpp runtime/Runtime.cpp
      PROPERTIES LANGUAGE CUDA)
    target_compile_options (CORSIKA8BackendRuntime PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    if (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "CUDA_OPENMP")
      target_compile_options (CORSIKA8BackendRuntime PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    endif ()
  elseif (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "HIP")
    enable_language (HIP)
    set_target_properties (CORSIKA8BackendRuntime PROPERTIES
      HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    set_target_properties (CORSIKA8KokkosRuntime PROPERTIES
      HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    set_source_files_properties (runtime/BackendRuntime.cpp runtime/Runtime.cpp
      PROPERTIES LANGUAGE HIP)
  endif ()
  list (APPEND CORSIKA_PRODUCTION_RUNTIME_TARGETS CORSIKA8BackendRuntime)
endif ()

if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TARGET IN LISTS CORSIKA_PRODUCTION_RUNTIME_TARGETS)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PUBLIC -fsanitize=address,undefined)
  endforeach ()
endif ()
install (TARGETS ${CORSIKA_PRODUCTION_RUNTIME_TARGETS}
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
