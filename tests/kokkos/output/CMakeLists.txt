# (c) Copyright 2026 CORSIKA Project. Distributed under the 3-clause BSD license.
add_executable (testProfileProjection testProfileProjection.cpp
  ProjectedStepRecordHeader.cpp ProfileProjectionViewHeader.cpp
  ProfileProjectionHeader.cpp LongitudinalCrossingsHeader.cpp)
target_link_libraries (testProfileProjection PRIVATE CORSIKA8ProfileProjection
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testProfileProjection PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testProfileProjection PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testProfileProjection PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testProfileProjection PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testProfileProjection PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testProfileProjection COMMAND testProfileProjection
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testProfileProjection.xml -r junit)
set_tests_properties (testProfileProjection PROPERTIES LABELS "kokkos;profile;output"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable (testProfileAccumulation testProfileAccumulation.cpp
  ProfileCountersHeader.cpp ProfileAccumulatorViewHeader.cpp
  ProfileAccumulationHeader.cpp ProfileAtomicOperationsHeader.cpp)
target_link_libraries (testProfileAccumulation PRIVATE CORSIKA8ProfileAccumulation
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testProfileAccumulation PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testProfileAccumulation PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testProfileAccumulation PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testProfileAccumulation PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testProfileAccumulation PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testProfileAccumulation COMMAND testProfileAccumulation
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testProfileAccumulation.xml -r junit)
set_tests_properties (testProfileAccumulation PROPERTIES LABELS "kokkos;profile;output"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable (testProfileStorage testProfileStorage.cpp
  ProfileProjectionConfigHeader.cpp
  ProfileResultHeader.cpp
  CooperativeEndpointHeader.cpp
  CheckedMemoryHeader.cpp
  ResidentMemoryBudgetHeader.cpp
  HostStagingSpaceHeader.cpp
  FixedAccumulatorMergeHeader.cpp
  CooperativeProfileMergeHeader.cpp
  HostProfileShardsHeader.cpp
  ProfileProjectionStorageHeader.cpp
  ProfileAccumulatorStorageHeader.cpp)
target_link_libraries (testProfileStorage PRIVATE CORSIKA8ProfileStorage
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testProfileStorage PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testProfileStorage PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testProfileStorage PRIVATE -fsanitize=address,undefined)
endif ()
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testProfileStorage PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testProfileStorage PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
add_test (NAME testProfileStorage COMMAND testProfileStorage
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testProfileStorage.xml -r junit)
set_tests_properties (testProfileStorage PROPERTIES LABELS "kokkos;profile;storage"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testProfileProjection testProfileProjection.cpp)
corsika_kokkos_device_test (testProfileAccumulation
  testProfileAccumulation.cpp
  ProfileAtomicOperationsHeader.cpp
)
corsika_kokkos_device_test (testProfileStorage
  testProfileStorage.cpp
  HostStagingSpaceHeader.cpp
  ProfileProjectionStorageHeader.cpp
  ProfileAccumulatorStorageHeader.cpp
)
