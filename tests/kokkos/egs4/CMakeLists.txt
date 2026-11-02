# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
# Distributed under the 3-clause BSD license; see LICENSE.

if (TARGET CORSIKA8NativeEgs4)
  add_executable (testEgs4Boundary testAtmosphere.cpp testMuonHandoff.cpp)
  target_link_libraries (testEgs4Boundary PRIVATE CORSIKA8NativeEgs4 Catch2::Catch2WithMain)
  add_test (NAME testEgs4Boundary COMMAND testEgs4Boundary)
  set_tests_properties (testEgs4Boundary PROPERTIES
    LABELS "kokkos;egs4;environment;muons" TIMEOUT 60)
  add_executable (testEgs4Tables testTables.cpp)
  add_executable (testEgs4Radiative testRadiative.cpp)
  target_link_libraries (testEgs4Radiative PRIVATE CORSIKA8NativeEgs4 Catch2::Catch2WithMain)
  target_compile_definitions (testEgs4Radiative PRIVATE
    CORSIKA_EGS4_TEST_TABLE="${CORSIKA_EGS4_TABLE}")
  target_link_libraries (testEgs4Tables PRIVATE CORSIKA8NativeEgs4 Catch2::Catch2WithMain)
  target_compile_definitions (testEgs4Tables PRIVATE
    CORSIKA_EGS4_TEST_TABLE="${CORSIKA_EGS4_TABLE}")
  if (CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "^CUDA")
    set_source_files_properties (testTables.cpp testRadiative.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties (testEgs4Tables PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options (testEgs4Tables PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties (testEgs4Radiative PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options (testEgs4Radiative PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  elseif (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "HIP")
    set_source_files_properties (testTables.cpp testRadiative.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties (testEgs4Tables testEgs4Radiative PROPERTIES
      HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  endif ()
  add_test (NAME testEgs4Tables COMMAND testEgs4Tables)
  add_test (NAME testEgs4Radiative COMMAND testEgs4Radiative)
  set_tests_properties (testEgs4Radiative PROPERTIES LABELS "kokkos;egs4;radiative"
    RUN_SERIAL TRUE TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
  set_tests_properties (testEgs4Tables PROPERTIES LABELS "kokkos;egs4;transport"
    RUN_SERIAL TRUE TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endif ()
