#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

function (corsika_add_continuous_test TARGET)
  add_executable (${TARGET} ${ARGN})
  target_link_libraries (${TARGET}
    PRIVATE CORSIKA8EmContinuous Catch2::Catch2WithMain)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES
    LABELS "kokkos;continuous"
    TIMEOUT 600
    RUN_SERIAL TRUE
    PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_CONTINUOUS_TEST_CACHE=${PROJECT_BINARY_DIR}/continuous-test-cache")
endfunction ()

corsika_add_continuous_test (testContinuousContracts testContinuousContracts.cpp
  ContinuousPhysicsQueriesHeader.cpp LeptonContinuousStepHeader.cpp InteractionRecordHeader.cpp
  FallbackRequestHeader.cpp TransportMassHeader.cpp)

if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  add_library (ContinuousTestFixture STATIC ContinuousFixture.cpp)
  target_link_libraries (ContinuousTestFixture PUBLIC CORSIKA8ProposalBridge PRIVATE CORSIKA8)
  # FIXME: the unmodified native NuclearComposition header returns const double.
  target_compile_options (ContinuousTestFixture PRIVATE -Wno-error=ignored-qualifiers)
  corsika_add_continuous_test (testContinuousQueries testContinuousQueries.cpp)
  corsika_add_continuous_test (testContinuousQueriesDevice testContinuousQueriesDevice.cpp)
  target_link_libraries (testContinuousQueries PRIVATE ContinuousTestFixture)
  target_link_libraries (testContinuousQueriesDevice
    PRIVATE ContinuousTestFixture CORSIKA8KokkosRuntime)
  if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
    target_compile_definitions (testContinuousQueriesDevice PRIVATE CORSIKA_EXPECT_CUDA)
  elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
    target_compile_definitions (testContinuousQueriesDevice PRIVATE CORSIKA_EXPECT_OPENMP)
  endif ()
endif ()

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testContinuousQueriesDevice testContinuousQueriesDevice.cpp)

corsika_kokkos_device_test (testContinuousContracts
  testContinuousContracts.cpp
)

corsika_kokkos_device_test (testContinuousQueries
  testContinuousQueries.cpp
)
