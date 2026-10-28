# (c) Copyright 2026 CORSIKA Project. Distributed under the 3-clause BSD license.
add_executable (testMoliereStep testMoliereStep.cpp MoliereStepHeader.cpp)
target_link_libraries (testMoliereStep PRIVATE CORSIKA8EmTransport
  CORSIKA8KokkosRuntime CORSIKA8MoliereInterpolation Catch2::Catch2WithMain)
target_compile_features (testMoliereStep PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testMoliereStep PRIVATE MoliereTestFixture)
  target_compile_definitions (testMoliereStep PRIVATE CORSIKA_MOLIERE_STAGE_REAL_TABLES)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testMoliereStep PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testMoliereStep PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testMoliereStep PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testMoliereStep PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testMoliereStep COMMAND testMoliereStep
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testMoliereStep.xml -r junit)
set_tests_properties (testMoliereStep PROPERTIES LABELS "kokkos;moliere-stage"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_MOLIERE_TEST_CACHE=${PROJECT_BINARY_DIR}/moliere-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/moliere-test-cache/auxiliary")

add_executable (testTransportRecords testTransportRecords.cpp
  PhotonTransportRecordHeader.cpp LeptonTransportRecordHeader.cpp
  NativeMuonPairCutRecordHeader.cpp)
target_link_libraries (testTransportRecords PRIVATE CORSIKA8EmSelection
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testTransportRecords PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testTransportRecords PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testTransportRecords PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testTransportRecords PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testTransportRecords PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testTransportRecords COMMAND testTransportRecords
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testTransportRecords.xml -r junit)
set_tests_properties (testTransportRecords PROPERTIES LABELS "kokkos;transport-records"
  TIMEOUT 120 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable (testLeptonTransport testLeptonTransport.cpp
  LeptonTransportStepHeader.cpp ../../environment/EnvironmentQueryAtmosphere.cpp)
target_link_libraries (testLeptonTransport PRIVATE CORSIKA8EmTransport
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testLeptonTransport PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testLeptonTransport PRIVATE ContinuousTestFixture)
  target_compile_definitions (testLeptonTransport PRIVATE CORSIKA_LEPTON_REAL_TABLES)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testLeptonTransport PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testLeptonTransport PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testLeptonTransport PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testLeptonTransport PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testLeptonTransport COMMAND testLeptonTransport
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testLeptonTransport.xml -r junit)
set_tests_properties (testLeptonTransport PROPERTIES LABELS "kokkos;lepton-transport"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_CONTINUOUS_TEST_CACHE=${PROJECT_BINARY_DIR}/continuous-test-cache")

add_executable (testPhotonTransport testPhotonTransport.cpp
  PhotonTransportStepHeader.cpp TransportGeometryConfigHeader.cpp
  ../../environment/EnvironmentQueryAtmosphere.cpp)
target_link_libraries (testPhotonTransport PRIVATE CORSIKA8EmTransport
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testPhotonTransport PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testPhotonTransport PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testPhotonTransport PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testPhotonTransport PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testPhotonTransport PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testPhotonTransport COMMAND testPhotonTransport
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testPhotonTransport.xml -r junit)
set_tests_properties (testPhotonTransport PROPERTIES LABELS "kokkos;photon-transport"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# A bounded real-table comparison of borrowed/managed/production environment access.
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  add_executable (testEnvironmentBorrowedTransport
    ../../environment/testEnvironmentBorrowedTransport.cpp
    ../../environment/EnvironmentQueryAtmosphere.cpp)
  target_link_libraries (testEnvironmentBorrowedTransport PRIVATE CORSIKA8EmTransport
    CORSIKA8EnvironmentStorage ContinuousTestFixture Catch2::Catch2WithMain)
  target_compile_features (testEnvironmentBorrowedTransport PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (testEnvironmentBorrowedTransport PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (testEnvironmentBorrowedTransport PRIVATE -fsanitize=address,undefined)
  endif ()
  if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
    target_compile_definitions (testEnvironmentBorrowedTransport PRIVATE CORSIKA_EXPECT_CUDA)
  elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
    target_compile_definitions (testEnvironmentBorrowedTransport PRIVATE CORSIKA_EXPECT_OPENMP)
  endif ()
  add_test (NAME testEnvironmentBorrowedTransport COMMAND testEnvironmentBorrowedTransport
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testEnvironmentBorrowedTransport.xml -r junit)
  set_tests_properties (testEnvironmentBorrowedTransport PROPERTIES
    LABELS "kokkos;environment;transport" TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_CONTINUOUS_TEST_CACHE=${PROJECT_BINARY_DIR}/continuous-test-cache")
endif ()

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testMoliereStep
  testMoliereStep.cpp
  MoliereStepHeader.cpp
)
corsika_kokkos_device_test (testTransportRecords testTransportRecords.cpp)
corsika_kokkos_device_test (testLeptonTransport
  testLeptonTransport.cpp
  LeptonTransportStepHeader.cpp
)
corsika_kokkos_device_test (testPhotonTransport
  testPhotonTransport.cpp
  PhotonTransportStepHeader.cpp
)
corsika_kokkos_device_test (testEnvironmentBorrowedTransport
  ../../environment/testEnvironmentBorrowedTransport.cpp
)
