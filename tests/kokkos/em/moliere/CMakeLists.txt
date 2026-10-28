#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

function (corsika_add_moliere_test TARGET)
  add_executable (${TARGET} ${ARGN})
  target_link_libraries (${TARGET} PRIVATE CORSIKA8EmModels Catch2::Catch2WithMain)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES
    LABELS "kokkos;moliere" TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_MOLIERE_TEST_CACHE=${PROJECT_BINARY_DIR}/moliere-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/moliere-test-cache/auxiliary")
endfunction ()

add_library (MoliereCacheReference STATIC MoliereCacheReference.cpp)
target_link_libraries (MoliereCacheReference PUBLIC CORSIKA8EmModels
  PRIVATE CubicInterpolation::CubicInterpolation)
corsika_add_moliere_test (testMoliereContracts testMoliereContracts.cpp
  MoliereScatteringHeader.cpp MoliereInterpolationHeader.cpp)
corsika_add_moliere_test (testMoliereCache testMoliereCache.cpp)
target_link_libraries (testMoliereCache PRIVATE CORSIKA8MoliereInterpolation MoliereCacheReference)

if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  add_library (MoliereTestFixture STATIC MoliereFixture.cpp
    ../models/AuxiliaryModelFixture.cpp ../models/AuxiliaryExportReference.cpp)
  target_link_libraries (MoliereTestFixture PUBLIC CORSIKA8ProposalBridge
    CORSIKA8MoliereInterpolation MoliereCacheReference PRIVATE CORSIKA8)
  corsika_add_moliere_test (testMoliereQueries testMoliereQueries.cpp)
  corsika_add_moliere_test (testMoliereQueriesDevice testMoliereQueriesDevice.cpp
    MoliereInterpolationStorageHeader.cpp)
  target_link_libraries (testMoliereQueries PRIVATE MoliereTestFixture)
  target_link_libraries (testMoliereQueriesDevice PRIVATE MoliereTestFixture CORSIKA8KokkosRuntime)
  if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
    target_compile_definitions (testMoliereQueriesDevice PRIVATE CORSIKA_EXPECT_CUDA)
  elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
    target_compile_definitions (testMoliereQueriesDevice PRIVATE CORSIKA_EXPECT_OPENMP)
  endif ()
endif ()

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testMoliereQueriesDevice
  testMoliereQueriesDevice.cpp
  MoliereInterpolationStorageHeader.cpp
)
