#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#
# Candidate implementation is always a real built/installed library.
if (TARGET ${BC_LIBRARY_PREFIX}NativeRequirements)
  add_executable (${BC_PREFIX}_snapshot_key
    "${BC_SESSION_TESTS}/SnapshotKey.cpp" "${BC_SESSION_TESTS}/SnapshotKeyHeader.cpp")
  target_link_libraries (${BC_PREFIX}_snapshot_key PRIVATE ${BC_NATIVE})
  target_include_directories (${BC_PREFIX}_snapshot_key PRIVATE "${BC_SESSION_REFERENCE}")
  target_compile_features (${BC_PREFIX}_snapshot_key PRIVATE cxx_std_17)
  add_test (NAME ${BC_PREFIX}_snapshot_key COMMAND ${BC_PREFIX}_snapshot_key)

  add_executable (${BC_PREFIX}_requirements
    "${BC_SESSION_TESTS}/Requirements.cpp" "${BC_SESSION_TESTS}/RequirementsHeader.cpp")
  target_link_libraries (${BC_PREFIX}_requirements PRIVATE ${BC_LIBRARY_PREFIX}NativeRequirements)
  target_include_directories (${BC_PREFIX}_requirements PRIVATE "${BC_SESSION_REFERENCE}")
  target_compile_features (${BC_PREFIX}_requirements PRIVATE cxx_std_17)
  add_test (NAME ${BC_PREFIX}_requirements COMMAND ${BC_PREFIX}_requirements)
endif ()

if (TARGET ${BC_LIBRARY_PREFIX}EmBackend AND BC_REAL_EXPORT)
  if (BC_INSTALLED)
    add_library (${BC_PREFIX}_real_tables STATIC
      "${BC_FIXTURE_ROOT}/tests/kokkos/em/tables/NativeTableFixture.cpp"
      "${BC_FIXTURE_ROOT}/tests/kokkos/em/tables/ProposalNativeExportReference.cpp"
      "${BC_FIXTURE_ROOT}/tests/kokkos/em/models/AuxiliaryModelFixture.cpp"
      "${BC_FIXTURE_ROOT}/tests/kokkos/em/models/AuxiliaryExportReference.cpp")
    target_link_libraries (${BC_PREFIX}_real_tables PUBLIC
      ${BC_NATIVE} ${BC_LIBRARY_PREFIX}ProposalBridge ${BC_LIBRARY_PREFIX}MoliereInterpolation)
    target_compile_features (${BC_PREFIX}_real_tables PRIVATE cxx_std_17)
    set (BC_REAL_TABLE_LIBRARIES ${BC_PREFIX}_real_tables)
  else ()
    set (BC_REAL_TABLE_LIBRARIES NativeTableTestFixture AuxiliaryModelTestFixture)
  endif ()

  set (BC_EXECUTIONS)
  if (BC_BACKEND STREQUAL "CUDA_OPENMP")
    set (BC_EXECUTIONS cuda openmp cuda-openmp openmp-cuda)
  elseif (BC_BACKEND STREQUAL "OPENMP")
    set (BC_EXECUTIONS openmp)
  elseif (BC_BACKEND STREQUAL "CUDA")
    set (BC_EXECUTIONS cuda)
  elseif (BC_BACKEND STREQUAL "HIP")
    set (BC_EXECUTIONS hip)
  elseif (BC_BACKEND STREQUAL "SYCL")
    set (BC_EXECUTIONS sycl)
  else ()
    message (FATAL_ERROR "A real EmBackend target has no recognized execution backend")
  endif ()
  foreach (NAME IN ITEMS Stages Router)
    add_executable (${BC_PREFIX}_backend_${NAME} "${BC_BACKEND_TESTS}/Backend${NAME}.cpp")
    target_link_libraries (${BC_PREFIX}_backend_${NAME} PRIVATE
      ${BC_LIBRARY_PREFIX}EmBackend ${BC_REAL_TABLE_LIBRARIES})
    target_include_directories (${BC_PREFIX}_backend_${NAME} PRIVATE "${BC_FIXTURE_ROOT}")
    target_compile_features (${BC_PREFIX}_backend_${NAME} PRIVATE cxx_std_17)
  endforeach ()
  target_sources (${BC_PREFIX}_backend_Stages PRIVATE "${BC_BACKEND_TESTS}/EnvironmentDeviceQueries.cpp")
  if (BC_BACKEND MATCHES "^CUDA")
    enable_language (CUDA)
    set_source_files_properties ("${BC_BACKEND_TESTS}/EnvironmentDeviceQueries.cpp" PROPERTIES LANGUAGE CUDA)
    set_target_properties (${BC_PREFIX}_backend_Stages PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    target_compile_options (${BC_PREFIX}_backend_Stages PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  elseif (BC_BACKEND STREQUAL "HIP")
    enable_language (HIP)
    set_source_files_properties ("${BC_BACKEND_TESTS}/EnvironmentDeviceQueries.cpp" PROPERTIES LANGUAGE HIP)
    set_target_properties (${BC_PREFIX}_backend_Stages PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  endif ()
  foreach (EXECUTION IN LISTS BC_EXECUTIONS)
    foreach (NAME IN ITEMS Stages Router)
      add_test (NAME ${BC_PREFIX}_backend_${NAME}_${EXECUTION}
        COMMAND ${BC_PREFIX}_backend_${NAME} ${EXECUTION}
        "${CMAKE_CURRENT_BINARY_DIR}/${BC_PREFIX}-native-cache")
      set_tests_properties (${BC_PREFIX}_backend_${NAME}_${EXECUTION} PROPERTIES
        RUN_SERIAL TRUE TIMEOUT 1200 ENVIRONMENT
        "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_AUXILIARY_TEST_CACHE=${CMAKE_CURRENT_BINARY_DIR}/${BC_PREFIX}-aux-cache")
    endforeach ()
  endforeach ()

  if (TARGET ${BC_LIBRARY_PREFIX}EmRunSession)
    add_executable (${BC_PREFIX}_session
      "${BC_SESSION_TESTS}/Session.cpp" "${BC_SESSION_TESTS}/SessionHeader.cpp"
      "${BC_SESSION_TESTS}/SessionPhysicsFixture.cpp")
    target_link_libraries (${BC_PREFIX}_session PRIVATE ${BC_LIBRARY_PREFIX}EmRunSession)
    target_include_directories (${BC_PREFIX}_session PRIVATE "${BC_FIXTURE_ROOT}")
    target_compile_features (${BC_PREFIX}_session PRIVATE cxx_std_17)
    foreach (EXECUTION IN LISTS BC_EXECUTIONS)
      foreach (SCENARIO IN ITEMS cold-hot environment early-failure late-failure)
        add_test (NAME ${BC_PREFIX}_session_${SCENARIO}_${EXECUTION}
          COMMAND ${BC_PREFIX}_session ${SCENARIO} ${EXECUTION}
          "${CMAKE_CURRENT_BINARY_DIR}/${BC_PREFIX}-session-cache")
        set_tests_properties (${BC_PREFIX}_session_${SCENARIO}_${EXECUTION} PROPERTIES
          RUN_SERIAL TRUE TIMEOUT 1800 ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
      endforeach ()
    endforeach ()
  endif ()
endif ()
