#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

# Context publication preserves production initialization and shower waits.
add_executable (testPhysicsContext testPhysicsContext.cpp
  PhysicsContextHeader.cpp PhysicsContextDataHeader.cpp PhysicsContextReport.cpp)
target_link_libraries (testPhysicsContext PRIVATE CORSIKA8PhysicsContext
  CORSIKA8KokkosEnvironmentSearch Catch2::Catch2WithMain spdlog::spdlog)
target_compile_features (testPhysicsContext PRIVATE cxx_std_17)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testPhysicsContext PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testPhysicsContext PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testPhysicsContext PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testPhysicsContext PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testPhysicsContext COMMAND testPhysicsContext
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testPhysicsContext.xml -r junit)
set_tests_properties (testPhysicsContext PROPERTIES LABELS "kokkos;context;lifetime"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# One-front output order and capacity checks, including real native tables when enabled.
add_executable (testPhotonFront testPhotonFront.cpp PhotonFrontHeader.cpp
  PhotonFrontDataHeader.cpp PhotonFrontWorkspaceHeader.cpp ObservationRecordHeader.cpp)
target_link_libraries (testPhotonFront PRIVATE CORSIKA8PhotonFront
  CORSIKA8NativeTableQueries CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testPhotonFront PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testPhotonFront PRIVATE NativeTableTestFixture AuxiliaryModelTestFixture)
  target_compile_definitions (testPhotonFront PRIVATE CORSIKA_TEST_FRONT_NATIVE)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testPhotonFront PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testPhotonFront PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testPhotonFront PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testPhotonFront PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testPhotonFront COMMAND testPhotonFront
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testPhotonFront.xml -r junit)
set_tests_properties (testPhotonFront PROPERTIES LABELS "kokkos;photon;front"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache")

# Photon cascade queue ownership, checkpoints and independent multi-front oracle.
add_executable (testPhotonCascade testPhotonCascade.cpp PhotonCascadeHeader.cpp
  PhotonCascadeResultHeader.cpp ProcessStatisticsHeader.cpp)
target_link_libraries (testPhotonCascade PRIVATE CORSIKA8PhotonCascade
  CORSIKA8NativeTableQueries CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testPhotonCascade PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testPhotonCascade PRIVATE NativeTableTestFixture AuxiliaryModelTestFixture)
  target_compile_definitions (testPhotonCascade PRIVATE CORSIKA_TEST_CASCADE_NATIVE)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testPhotonCascade PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testPhotonCascade PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testPhotonCascade PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testPhotonCascade PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testPhotonCascade COMMAND testPhotonCascade
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testPhotonCascade.xml -r junit)
set_tests_properties (testPhotonCascade PROPERTIES LABELS "kokkos;photon;cascade"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache")

# Lepton front capacities, stable scans and independent per-particle oracle.
add_executable (testLeptonFront testLeptonFront.cpp LeptonFrontHeader.cpp
  LeptonFrontDataHeader.cpp LeptonFrontWorkspaceHeader.cpp)
target_link_libraries (testLeptonFront PRIVATE CORSIKA8LeptonFront
  CORSIKA8NativeTableQueries CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testLeptonFront PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testLeptonFront PRIVATE NativeTableTestFixture AuxiliaryModelTestFixture MoliereTestFixture)
  target_compile_definitions (testLeptonFront PRIVATE CORSIKA_TEST_FRONT_NATIVE)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testLeptonFront PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testLeptonFront PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testLeptonFront PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testLeptonFront PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testLeptonFront COMMAND testLeptonFront
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testLeptonFront.xml -r junit)
set_tests_properties (testLeptonFront PROPERTIES LABELS "kokkos;lepton;front"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache;CORSIKA_MOLIERE_TEST_CACHE=${PROJECT_BINARY_DIR}/moliere-test-cache")

# Distinct source/packed domains, cross-front sums/maxima and host profile shards.
add_executable (testLeptonStatistics testLeptonStatistics.cpp LeptonStatisticsHeader.cpp
  testMaterialBoundaryStatistics.cpp MaterialBoundaryFixture.cpp)
target_link_libraries (testLeptonStatistics PRIVATE CORSIKA8LeptonStatistics
  CORSIKA8KokkosEnvironment Catch2::Catch2WithMain)
set_source_files_properties (MaterialBoundaryFixture.cpp PROPERTIES LANGUAGE CXX)
if (CMAKE_CUDA_COMPILER_LOADED)
  set_source_files_properties (testMaterialBoundaryStatistics.cpp PROPERTIES LANGUAGE CUDA)
endif ()
target_compile_features (testLeptonStatistics PRIVATE cxx_std_17)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testLeptonStatistics PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testLeptonStatistics PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testLeptonStatistics PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testLeptonStatistics PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testLeptonStatistics COMMAND testLeptonStatistics
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testLeptonStatistics.xml -r junit)
set_tests_properties (testLeptonStatistics PROPERTIES LABELS "kokkos;lepton;statistics"
  TIMEOUT 120 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# Lepton cascade queue ownership, checkpoints and independent multi-front oracle.
add_executable (testLeptonCascade testLeptonCascade.cpp LeptonCascadeHeader.cpp
  LeptonCascadeResultHeader.cpp)
target_link_libraries (testLeptonCascade PRIVATE CORSIKA8LeptonCascade
  CORSIKA8NativeTableQueries CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testLeptonCascade PRIVATE cxx_std_17)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  target_link_libraries (testLeptonCascade PRIVATE NativeTableTestFixture AuxiliaryModelTestFixture MoliereTestFixture)
  target_compile_definitions (testLeptonCascade PRIVATE CORSIKA_TEST_CASCADE_NATIVE)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testLeptonCascade PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testLeptonCascade PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testLeptonCascade PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testLeptonCascade PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testLeptonCascade COMMAND testLeptonCascade
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testLeptonCascade.xml -r junit)
set_tests_properties (testLeptonCascade PROPERTIES LABELS "kokkos;lepton;cascade"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache;CORSIKA_AUXILIARY_TEST_CACHE=${PROJECT_BINARY_DIR}/auxiliary-model-test-cache;CORSIKA_MOLIERE_TEST_CACHE=${PROJECT_BINARY_DIR}/moliere-test-cache")

add_subdirectory (backend)

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testPhysicsContext
  testPhysicsContext.cpp
  PhysicsContextHeader.cpp
)
corsika_kokkos_device_test (testPhotonFront
  testPhotonFront.cpp
  PhotonFrontHeader.cpp
  PhotonFrontWorkspaceHeader.cpp
)
corsika_kokkos_device_test (testPhotonCascade
  testPhotonCascade.cpp
  PhotonCascadeHeader.cpp
)
corsika_kokkos_device_test (testLeptonFront
  testLeptonFront.cpp
  LeptonFrontHeader.cpp
  LeptonFrontWorkspaceHeader.cpp
)
corsika_kokkos_device_test (testLeptonStatistics
  testLeptonStatistics.cpp
  LeptonStatisticsHeader.cpp
  testMaterialBoundaryStatistics.cpp
)
corsika_kokkos_device_test (testLeptonCascade
  testLeptonCascade.cpp
  LeptonCascadeHeader.cpp
)
