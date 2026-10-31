#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

add_executable (testWavefrontBucketing testWavefrontBucketing.cpp WavefrontBucketingHeader.cpp)
target_link_libraries (testWavefrontBucketing PRIVATE
  CORSIKA8WavefrontBucketing Catch2::Catch2WithMain)
target_compile_features (testWavefrontBucketing PRIVATE cxx_std_17)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testWavefrontBucketing PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testWavefrontBucketing PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testWavefrontBucketing PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testWavefrontBucketing PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testWavefrontBucketing COMMAND testWavefrontBucketing
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testWavefrontBucketing.xml -r junit)
set_tests_properties (testWavefrontBucketing PROPERTIES LABELS "kokkos;scheduling;bucketing"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# These host scheduling checks do not initialize Kokkos or start worker processes.
add_executable (testCpuParticleScheduler testCpuParticleScheduler.cpp CpuParticleSchedulerHeader.cpp)
add_executable (testProcessSequenceCompatibility testProcessSequenceCompatibility.cpp ProcessSequenceCompatibilityHeader.cpp)
add_executable (testScheduledFallback testScheduledFallback.cpp)
find_package (nlohmann_json CONFIG REQUIRED)
add_executable (testFrontierDirections testFrontierDirections.cpp)
foreach (TARGET testCpuParticleScheduler testProcessSequenceCompatibility testScheduledFallback testFrontierDirections)
  target_link_libraries (${TARGET} PRIVATE CORSIKA8 Catch2::Catch2WithMain)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES LABELS "kokkos;scheduling;host"
    TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 1)
endforeach ()
target_link_libraries (testScheduledFallback PRIVATE CORSIKA8EmStochastic)
target_link_libraries (testFrontierDirections PRIVATE nlohmann_json::nlohmann_json)

# Host-side wire/capture contracts for isolated hadronic final-state workers.
find_package (Threads REQUIRED)
add_executable (testHadronicBatchProtocol testHadronicBatchProtocol.cpp HadronicBatchProtocolHeader.cpp)
add_executable (testHadronicInteractionDeferral testHadronicInteractionDeferral.cpp HadronicInteractionDeferralHeader.cpp)
foreach (TARGET testHadronicBatchProtocol testHadronicInteractionDeferral)
  target_link_libraries (${TARGET} PRIVATE CORSIKA8 Catch2::Catch2WithMain Threads::Threads)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES LABELS "kokkos;scheduling;hadronic;host"
    TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 1)
endforeach ()

# Host-side scalar transport; no device execution or process workers are started.
add_executable (testScalarCascadeStepper testScalarCascadeStepper.cpp
  testDeferredScalarStep.cpp testScalarStateContracts.cpp ScalarCascadeStepperHeader.cpp)
target_link_libraries (testScalarCascadeStepper PRIVATE CORSIKA8 Catch2::Catch2WithMain)
target_compile_features (testScalarCascadeStepper PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testScalarCascadeStepper PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testScalarCascadeStepper PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testScalarCascadeStepper COMMAND testScalarCascadeStepper
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testScalarCascadeStepper.xml -r junit)
set_tests_properties (testScalarCascadeStepper PROPERTIES
  LABELS "kokkos;scheduling;scalar;host" TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 1)

# Deterministic queue planning and process-isolated wire handling.
add_executable (testHadronicWorkQueue testHadronicWorkQueue.cpp
  testHadronicQueueReference.cpp HadronicWorkQueueHeader.cpp)
set (HADRONIC_SCHEDULING_TESTS testHadronicWorkQueue)
if (UNIX)
  add_executable (hadronicProtocolTestWorker HadronicProtocolTestWorker.cpp)
  target_link_libraries (hadronicProtocolTestWorker PRIVATE CORSIKA8 Threads::Threads)
  target_compile_features (hadronicProtocolTestWorker PRIVATE cxx_std_17)
  add_executable (testHadronicProcessPool testHadronicProcessPool.cpp HadronicProcessPoolHeader.cpp)
  target_compile_definitions (testHadronicProcessPool PRIVATE
    HADRONIC_PROTOCOL_TEST_WORKER="$<TARGET_FILE:hadronicProtocolTestWorker>")
  add_dependencies (testHadronicProcessPool hadronicProtocolTestWorker)
  list (APPEND HADRONIC_SCHEDULING_TESTS testHadronicProcessPool)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (hadronicProtocolTestWorker PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (hadronicProtocolTestWorker PRIVATE -fsanitize=address,undefined)
  endif ()
endif ()
foreach (TARGET ${HADRONIC_SCHEDULING_TESTS})
  target_link_libraries (${TARGET} PRIVATE CORSIKA8 Catch2::Catch2WithMain Threads::Threads)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES LABELS "kokkos;scheduling;hadronic;host"
    TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 1)
endforeach ()

if (NOT "$CACHE{C8_FLUKA_ENABLED}" STREQUAL "")
  add_subdirectory (fluka)
endif ()

# Explicit final-state deferral and its host-side coordinating loop.
add_subdirectory (deferral)
if (UNIX)
  add_subdirectory (hybrid)
endif ()

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testWavefrontBucketing
  testWavefrontBucketing.cpp
  WavefrontBucketingHeader.cpp
)
