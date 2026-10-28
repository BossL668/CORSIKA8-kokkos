# Shared registration for the receiver and a prefix-only installed consumer.
foreach (NAME PREFIX TESTS REFERENCE)
  if (NOT DEFINED CORSIKA_RUNTIME_CONTRACT_${NAME})
    message (FATAL_ERROR "Set CORSIKA_RUNTIME_CONTRACT_${NAME} before including this file")
  endif ()
endforeach ()
if (NOT TARGET CORSIKA8::TuningCache)
  message (FATAL_ERROR "The selected installation/configuration has no TuningCache target")
endif ()
set (PREFIX "${CORSIKA_RUNTIME_CONTRACT_PREFIX}")
set (TESTS "${CORSIKA_RUNTIME_CONTRACT_TESTS}")
set (REFERENCE "${CORSIKA_RUNTIME_CONTRACT_REFERENCE}")
foreach (FILE
    "${TESTS}/RuntimeContract.cpp" "${TESTS}/RuntimeHeader.cpp"
    "${TESTS}/CacheContract.cpp" "${TESTS}/CacheHeader.cpp"
    "${REFERENCE}/src/accelerator/em/kokkos/KokkosTuningCache.cpp"
    "${REFERENCE}/src/accelerator/em/tables/Sha256.cpp")
  if (NOT EXISTS "${FILE}")
    message (FATAL_ERROR "Missing frozen contract input: ${FILE}")
  endif ()
endforeach ()

add_executable (${PREFIX}_cache
  "${TESTS}/CacheContract.cpp" "${TESTS}/CacheHeader.cpp"
  "${REFERENCE}/src/accelerator/em/kokkos/KokkosTuningCache.cpp"
  "${REFERENCE}/src/accelerator/em/tables/Sha256.cpp")
target_link_libraries (${PREFIX}_cache PRIVATE CORSIKA8::TuningCache)
# Only the untouched legacy oracle is a test include directory. Candidate headers
# come from the linked target (the installation prefix in the consumer).
target_include_directories (${PREFIX}_cache PRIVATE "${REFERENCE}")
target_compile_features (${PREFIX}_cache PRIVATE cxx_std_17)
add_test (NAME ${PREFIX}_cache
  COMMAND ${PREFIX}_cache "${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}-cache-data")
set (CONTRACT_TEST_NAMES ${PREFIX}_cache)
set (CONTRACT_TARGET_NAMES ${PREFIX}_cache)

if (TARGET CORSIKA8::BackendRuntime)
  get_target_property (DEFINITIONS CORSIKA8::BackendRuntime INTERFACE_COMPILE_DEFINITIONS)
  set (BACKENDS "")
  foreach (DEFINITION IN LISTS DEFINITIONS)
    if (DEFINITION MATCHES "^CORSIKA_KOKKOS_BACKEND_([A-Z_]+)(=1)?$")
      list (APPEND BACKENDS "${CMAKE_MATCH_1}")
    endif ()
  endforeach ()
  list (LENGTH BACKENDS BACKEND_COUNT)
  if (NOT BACKEND_COUNT EQUAL 1)
    message (FATAL_ERROR "BackendRuntime must export exactly one production backend")
  endif ()
  list (GET BACKENDS 0 BACKEND)
  if (NOT BACKEND MATCHES "^(OPENMP|CUDA|HIP|SYCL|CUDA_OPENMP)$")
    message (FATAL_ERROR "Unsupported exported production backend: ${BACKEND}")
  endif ()
  string (TOLOWER "${BACKEND}" EXECUTION)
  if (BACKEND STREQUAL "CUDA_OPENMP")
    set (EXECUTION cuda)
  endif ()
  if (BACKEND MATCHES "^CUDA")
    enable_language (CUDA)
    set_source_files_properties ("${TESTS}/RuntimeContract.cpp" PROPERTIES LANGUAGE CUDA)
  elseif (BACKEND STREQUAL "HIP")
    enable_language (HIP)
    set_source_files_properties ("${TESTS}/RuntimeContract.cpp" PROPERTIES LANGUAGE HIP)
  endif ()
  add_executable (${PREFIX}_runtime
    "${TESTS}/RuntimeContract.cpp" "${TESTS}/RuntimeHeader.cpp")
  target_link_libraries (${PREFIX}_runtime PRIVATE
    CORSIKA8::BackendRuntime CORSIKA8::KokkosRuntime)
  target_compile_features (${PREFIX}_runtime PRIVATE cxx_std_17)
  if (BACKEND MATCHES "^CUDA")
    set_target_properties (${PREFIX}_runtime PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    target_compile_options (${PREFIX}_runtime PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  elseif (BACKEND STREQUAL "HIP")
    set_target_properties (${PREFIX}_runtime PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  endif ()
  foreach (SCENARIO owner generic-first external-first unavailable cooperative)
    add_test (NAME ${PREFIX}_runtime_${SCENARIO}
      COMMAND ${PREFIX}_runtime ${SCENARIO} ${EXECUTION} 1)
    list (APPEND CONTRACT_TEST_NAMES ${PREFIX}_runtime_${SCENARIO})
  endforeach ()
  if (BACKEND STREQUAL "OPENMP")
    add_test (NAME ${PREFIX}_runtime_threads COMMAND ${PREFIX}_runtime owner openmp 2)
    list (APPEND CONTRACT_TEST_NAMES ${PREFIX}_runtime_threads)
  else ()
    add_test (NAME ${PREFIX}_runtime_gpu-threads
      COMMAND ${PREFIX}_runtime gpu-threads ${EXECUTION})
    list (APPEND CONTRACT_TEST_NAMES ${PREFIX}_runtime_gpu-threads)
  endif ()
  if (BACKEND STREQUAL "CUDA_OPENMP")
    add_test (NAME ${PREFIX}_runtime_dual-openmp COMMAND ${PREFIX}_runtime owner openmp 2)
    list (APPEND CONTRACT_TEST_NAMES ${PREFIX}_runtime_dual-openmp)
  endif ()
  list (APPEND CONTRACT_TARGET_NAMES ${PREFIX}_runtime)
else ()
  message (STATUS "No production BackendRuntime target; cache contract still runs")
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TARGET IN LISTS CONTRACT_TARGET_NAMES)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    target_link_options (${TARGET} PRIVATE -fsanitize=address,undefined)
  endforeach ()
endif ()
set_tests_properties (${CONTRACT_TEST_NAMES} PROPERTIES
  LABELS "kokkos;runtime;production-reference" RUN_SERIAL TRUE TIMEOUT 180
  ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
