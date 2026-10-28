#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

# Avoid linking the full physics target in runtime and data-boundary tests.
find_package (Catch2 REQUIRED)
find_package (Threads REQUIRED)

# Sources using CUDA Kokkos headers require the device compiler. Native model
# fixtures and the air-shower application retain their ordinary C++ compiler.
function (corsika_kokkos_device_test TARGET)
  if (TARGET ${TARGET} AND Kokkos_DEVICES MATCHES "CUDA")
    set_source_files_properties (${ARGN} PROPERTIES LANGUAGE CUDA)
    set_target_properties (${TARGET} PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    target_compile_options (${TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_link_libraries (${TARGET} PRIVATE CUDA::cudart)
  endif ()
endfunction ()

add_executable (testKokkosRuntime runtime/testRuntime.cpp)
target_link_libraries (testKokkosRuntime
  PRIVATE CORSIKA8KokkosRuntime Catch2::Catch2WithMain Threads::Threads
)
target_compile_features (testKokkosRuntime PRIVATE cxx_std_17)
corsika_kokkos_device_test (testKokkosRuntime runtime/testRuntime.cpp)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosRuntime PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
endif ()

# Each lifecycle scenario needs its own process: Kokkos cannot reinitialize.
set (KOKKOS_RUNTIME_SCENARIOS
  owned
  shared
  explicit_configuration
  conflicting_configuration
  borrowed
  invalid_options
  finalized
  coordinator_thread
  exception_cleanup
  external_configuration
  explicit_execution
)

file (MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_outputs)
foreach (SCENARIO IN LISTS KOKKOS_RUNTIME_SCENARIOS)
  set (TEST_NAME testKokkosRuntime_${SCENARIO})
  add_test (NAME ${TEST_NAME}
    COMMAND testKokkosRuntime "[runtime_${SCENARIO}]"
      -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TEST_NAME}.xml -r junit
  )
  set_tests_properties (${TEST_NAME} PROPERTIES
    LABELS "kokkos;runtime"
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
    PROCESSORS 2
    RUN_SERIAL TRUE
    TIMEOUT 60
  )
endforeach ()

add_subdirectory (data)
add_subdirectory (environment)

add_subdirectory (utility)
add_subdirectory (em/tables)
add_subdirectory (em/models)
add_subdirectory (em/moliere)
add_subdirectory (em/continuous)
add_subdirectory (em/discrete)
add_subdirectory (em/stochastic)
add_subdirectory (em/selection)
add_subdirectory (em/transport)
add_subdirectory (em/finalstate)
add_subdirectory (output)

add_subdirectory (scheduling)
add_subdirectory (runtime)
add_subdirectory (em/execution)

add_subdirectory (radio)

add_subdirectory (em/proposal)

add_subdirectory (diagnostics)

add_subdirectory (application)
