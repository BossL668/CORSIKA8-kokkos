#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#
option (CORSIKA_TEST_COORDINATOR_PROCESS
  "Run real process failure/timeout tests; requires nvidia-smi and NVIDIA devices 0,1" OFF)
find_package (OpenSSL REQUIRED)
find_package (nlohmann_json CONFIG REQUIRED)
find_package (Python3 REQUIRED COMPONENTS Interpreter)
add_library (${BC_PREFIX}_coordinator_header OBJECT "${BC_TESTS}/Header.cpp")
target_link_libraries (${BC_PREFIX}_coordinator_header PRIVATE
  ${BC_NATIVE} nlohmann_json::nlohmann_json)
target_compile_features (${BC_PREFIX}_coordinator_header PRIVATE cxx_std_17)
foreach (MODE IN ITEMS candidate production)
  add_executable (${BC_PREFIX}_coordinator_${MODE} "${BC_TESTS}/CoordinatorProbe.cpp")
  target_compile_features (${BC_PREFIX}_coordinator_${MODE} PRIVATE cxx_std_17)
  target_link_libraries (${BC_PREFIX}_coordinator_${MODE} PRIVATE
    ${BC_NATIVE} nlohmann_json::nlohmann_json OpenSSL::Crypto)
  if (MODE STREQUAL "production")
    target_compile_definitions (${BC_PREFIX}_coordinator_${MODE} PRIVATE PRODUCTION_REFERENCE)
    target_include_directories (${BC_PREFIX}_coordinator_${MODE} BEFORE PRIVATE "${BC_REFERENCE}")
    target_sources (${BC_PREFIX}_coordinator_${MODE} PRIVATE
      "${BC_REFERENCE}/applications/detail/air_shower_multigpu/NativeCoordinator.cpp"
      "${BC_REFERENCE}/applications/detail/air_shower_multigpu/NativeIO.cpp")
  elseif (TARGET ${BC_COORDINATOR})
    target_link_libraries (${BC_PREFIX}_coordinator_${MODE} PRIVATE ${BC_COORDINATOR})
    target_sources (${BC_PREFIX}_coordinator_${MODE} PRIVATE
      $<TARGET_OBJECTS:${BC_PREFIX}_coordinator_header>)
  elseif (NOT BC_INSTALLED)
    # Source-only host compilation is legal in OFF; none of these headers are installed.
    target_sources (${BC_PREFIX}_coordinator_${MODE} PRIVATE
      "${PROJECT_SOURCE_DIR}/src/kokkos/scheduling/AirShowerCoordinator.cpp"
      "${PROJECT_SOURCE_DIR}/src/kokkos/scheduling/StaticFrontierIO.cpp"
      "${PROJECT_SOURCE_DIR}/src/kokkos/application/DeviceSelection.cpp"
      $<TARGET_OBJECTS:${BC_PREFIX}_coordinator_header>)
  else ()
    message (FATAL_ERROR "Installed coordinator tests require the exported coordinator target")
  endif ()
endforeach ()
set (BC_COORDINATOR_CASES io cli firn)
if (CORSIKA_TEST_COORDINATOR_PROCESS)
  add_executable (${BC_PREFIX}_coordinator_sleep_child "${BC_TESTS}/SleepChild.cpp")
  # The preserved Python fixture locates this basename next to its candidate binary.
  set_target_properties (${BC_PREFIX}_coordinator_sleep_child PROPERTIES
    OUTPUT_NAME coordinator_sleep_child)
  list (APPEND BC_COORDINATOR_CASES process)
endif ()
foreach (CASE IN LISTS BC_COORDINATOR_CASES)
  add_test (NAME ${BC_PREFIX}_coordinator_${CASE} COMMAND Python3::Interpreter
    "${BC_TESTS}/compare.py"
    --candidate $<TARGET_FILE:${BC_PREFIX}_coordinator_candidate>
    --production $<TARGET_FILE:${BC_PREFIX}_coordinator_production>
    --mode ${CASE} --work "${CMAKE_CURRENT_BINARY_DIR}/${BC_PREFIX}-coordinator-${CASE}")
  set_tests_properties (${BC_PREFIX}_coordinator_${CASE} PROPERTIES TIMEOUT 300)
endforeach ()
