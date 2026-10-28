#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

function (corsika_add_discrete_test TARGET)
  add_executable (${TARGET} ${ARGN})
  target_link_libraries (${TARGET} PRIVATE CORSIKA8EmDiscrete Catch2::Catch2WithMain)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES LABELS "kokkos;discrete"
    TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache")
endfunction ()

corsika_add_discrete_test (testDiscreteContracts testDiscreteContracts.cpp
  DiscretePhysicsQueriesHeader.cpp TableQueryResultHeader.cpp PhotonPairKinematicsHeader.cpp
  ProcessIdHeader.cpp)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  corsika_add_discrete_test (testDiscreteQueries testDiscreteQueries.cpp)
  corsika_add_discrete_test (testDiscreteQueriesDevice testDiscreteQueriesDevice.cpp)
  target_link_libraries (testDiscreteQueries PRIVATE NativeTableTestFixture)
  target_link_libraries (testDiscreteQueriesDevice PRIVATE NativeTableTestFixture CORSIKA8KokkosRuntime)
  if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
    target_compile_definitions (testDiscreteQueriesDevice PRIVATE CORSIKA_EXPECT_CUDA)
  elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
    target_compile_definitions (testDiscreteQueriesDevice PRIVATE CORSIKA_EXPECT_OPENMP)
  endif ()
endif ()

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testDiscreteQueriesDevice testDiscreteQueriesDevice.cpp)

corsika_kokkos_device_test (testDiscreteContracts
  testDiscreteContracts.cpp
)

corsika_kokkos_device_test (testDiscreteQueries
  testDiscreteQueries.cpp
)
