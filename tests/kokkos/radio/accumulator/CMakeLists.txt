# Preserve the production accumulation branches and compare a frozen oracle.
add_executable (testRadioAccumulator
  testRadioAccumulator.cpp
  testRadioMerge.cpp
  testHomogeneousRadio.cpp
  testHostRadioStaging.cpp
  testRadioCounters.cpp
  RadioAccumulatorHeader.cpp
  RadioAccumulatorOperationsHeader.cpp
  FixedRadioSnapshotHeader.cpp
  CooperativeRadioMergeHeader.cpp
  TuningParametersHeader.cpp)
target_link_libraries (testRadioAccumulator PRIVATE CORSIKA8RadioAccumulation
  Catch2::Catch2WithMain)
target_compile_features (testRadioAccumulator PRIVATE cxx_std_17)
if (CMAKE_CUDA_COMPILER_LOADED)
  target_compile_definitions (testRadioAccumulator PRIVATE CORSIKA_RADIO_EXPECT_CUDA)
  set_source_files_properties (testRadioAccumulator.cpp testRadioMerge.cpp
    testHomogeneousRadio.cpp testHostRadioStaging.cpp testRadioCounters.cpp
    RadioAccumulatorHeader.cpp RadioAccumulatorOperationsHeader.cpp
    PROPERTIES LANGUAGE CUDA)
else ()
  target_compile_definitions (testRadioAccumulator PRIVATE CORSIKA_RADIO_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testRadioAccumulator PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testRadioAccumulator PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testRadioAccumulator COMMAND testRadioAccumulator
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testRadioAccumulator.xml -r junit)
set_tests_properties (testRadioAccumulator PROPERTIES LABELS "kokkos;radio;accumulator"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testRadioAccumulator testRadioAccumulator.cpp)
corsika_kokkos_device_test (testRadioMerge testRadioMerge.cpp)
corsika_kokkos_device_test (testHomogeneousRadio testHomogeneousRadio.cpp)
corsika_kokkos_device_test (testHostRadioStaging testHostRadioStaging.cpp)
corsika_kokkos_device_test (testRadioCounters testRadioCounters.cpp)
