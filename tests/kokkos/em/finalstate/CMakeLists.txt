# (c) Copyright 2026 CORSIKA Project. Distributed under the 3-clause BSD license.
add_executable (testFinalStateContracts testFinalStateContracts.cpp
  FirstInteractionRecordHeader.cpp PhotonFinalStateRecordHeader.cpp
  LeptonFinalStateRecordHeader.cpp PhotonPairLpmSuppressionRecordHeader.cpp
  BremsLpmSuppressionRecordHeader.cpp EmThinningHeader.cpp)
target_link_libraries (testFinalStateContracts PRIVATE CORSIKA8KokkosData
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testFinalStateContracts PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testFinalStateContracts PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testFinalStateContracts PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testFinalStateContracts PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testFinalStateContracts PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testFinalStateContracts COMMAND testFinalStateContracts
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testFinalStateContracts.xml -r junit)
set_tests_properties (testFinalStateContracts PROPERTIES LABELS "kokkos;finalstate"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable (testPhotonFinalState testPhotonFinalState.cpp
  PhotonFinalStateStepHeader.cpp PhotonFinalStateRandomDomainsHeader.cpp)
target_link_libraries (testPhotonFinalState PRIVATE CORSIKA8EmFinalState
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testPhotonFinalState PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testPhotonFinalState PRIVATE NativeTableTestFixture AuxiliaryModelTestFixture)
  target_compile_definitions (testPhotonFinalState PRIVATE CORSIKA_TEST_PHOTON_NATIVE)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testPhotonFinalState PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testPhotonFinalState PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testPhotonFinalState PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testPhotonFinalState PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testPhotonFinalState COMMAND testPhotonFinalState
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testPhotonFinalState.xml -r junit)
set_tests_properties (testPhotonFinalState PROPERTIES LABELS "kokkos;finalstate;photon"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache")

add_executable (testLeptonFinalState testLeptonFinalState.cpp
  LeptonFinalStateStepHeader.cpp LeptonFinalStateRandomDomainsHeader.cpp
  PositronAtRestHeader.cpp BremsLpmPreparedSnapshotHeader.cpp)
target_link_libraries (testLeptonFinalState PRIVATE CORSIKA8EmFinalState
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testLeptonFinalState PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testLeptonFinalState PRIVATE NativeTableTestFixture AuxiliaryModelTestFixture)
  target_compile_definitions (testLeptonFinalState PRIVATE CORSIKA_TEST_LEPTON_NATIVE)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testLeptonFinalState PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testLeptonFinalState PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testLeptonFinalState PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testLeptonFinalState PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testLeptonFinalState COMMAND testLeptonFinalState
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testLeptonFinalState.xml -r junit)
set_tests_properties (testLeptonFinalState PROPERTIES LABELS "kokkos;finalstate;lepton"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache")

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testFinalStateContracts testFinalStateContracts.cpp)
corsika_kokkos_device_test (testPhotonFinalState testPhotonFinalState.cpp)
corsika_kokkos_device_test (testLeptonFinalState testLeptonFinalState.cpp)
