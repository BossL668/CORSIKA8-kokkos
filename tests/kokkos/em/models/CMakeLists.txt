#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

function (corsika_add_auxiliary_model_test TARGET)
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
    LABELS "kokkos;auxiliary_models" TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache")
endfunction ()

corsika_add_auxiliary_model_test (testAuxiliaryModelContracts
  testAuxiliaryModelContracts.cpp PhotonPairLpmHeader.cpp BremsLpmHeader.cpp
  MoliereSnapshotHeader.cpp ProposalNativeAuxHeader.cpp)

if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  add_library (AuxiliaryModelTestFixture STATIC AuxiliaryModelFixture.cpp
    AuxiliaryExportReference.cpp)
  target_link_libraries (AuxiliaryModelTestFixture
    PUBLIC CORSIKA8ProposalBridge PRIVATE CORSIKA8)
  corsika_add_auxiliary_model_test (testProposalNativeAux testProposalNativeAux.cpp)
  corsika_add_auxiliary_model_test (testAuxiliaryModelQueries testAuxiliaryModelQueries.cpp)
  corsika_add_auxiliary_model_test (testAuxiliaryModelQueriesDevice
    testAuxiliaryModelQueriesDevice.cpp)
  foreach (TARGET testProposalNativeAux testAuxiliaryModelQueries testAuxiliaryModelQueriesDevice)
    target_link_libraries (${TARGET} PRIVATE AuxiliaryModelTestFixture)
  endforeach ()
  target_link_libraries (testAuxiliaryModelQueriesDevice PRIVATE CORSIKA8KokkosRuntime)
  if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
    target_compile_definitions (testAuxiliaryModelQueriesDevice PRIVATE CORSIKA_EXPECT_CUDA)
  elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
    target_compile_definitions (testAuxiliaryModelQueriesDevice PRIVATE CORSIKA_EXPECT_OPENMP)
  endif ()
endif ()

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testAuxiliaryModelQueriesDevice
  testAuxiliaryModelQueriesDevice.cpp
)
