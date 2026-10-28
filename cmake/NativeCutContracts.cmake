#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#
# The native-only installed mode excludes only the uninstalled Kokkos forwarding API.
add_library (${BC_PREFIX}_cut_headers OBJECT
  "${BC_TESTS}/HeaderParticleCut.cpp" "${BC_TESTS}/HeaderPositronAtRest.cpp")
target_link_libraries (${BC_PREFIX}_cut_headers PRIVATE ${BC_NATIVE})
target_compile_features (${BC_PREFIX}_cut_headers PRIVATE cxx_std_17)
foreach (PAIR IN ITEMS
    "compare|CompareParticleCut.cpp" "readonly|ReadOnlyCut.cpp"
    "direction|PositronDirection.cpp" "production_regression|ProductionRegression.cpp")
  string (REPLACE "|" ";" FIELDS "${PAIR}")
  list (GET FIELDS 0 NAME)
  list (GET FIELDS 1 SOURCE)
  set (TARGET ${BC_PREFIX}_cut_${NAME})
  add_executable (${TARGET} "${BC_TESTS}/${SOURCE}")
  target_link_libraries (${TARGET} PRIVATE ${BC_NATIVE})
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  target_include_directories (${TARGET} PRIVATE
    "${BC_TEST_HELPERS}" "${BC_TEST_HELPERS}/../..")
  target_sources (${TARGET} PRIVATE $<TARGET_OBJECTS:${BC_PREFIX}_cut_headers>)
  if (BC_CUT_NATIVE_ONLY)
    target_compile_definitions (${TARGET} PRIVATE CORSIKA_CUT_NATIVE_ONLY)
  endif ()
  if (NAME STREQUAL "production_regression")
    target_link_libraries (${TARGET} PRIVATE Catch2::Catch2WithMain)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET})
  set_tests_properties (${TARGET} PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach ()
