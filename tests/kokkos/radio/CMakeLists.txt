# Production projection, an independent production oracle and actual C8 radio modules.
add_executable (testRadioProjection testRadioProjection.cpp RadioFixture.cpp
  FlatAtmosphereRadioSnapshotHeader.cpp
  RadioAtomicOperationsHeader.cpp
  RadioConfigHeader.cpp
  RadioObserverSnapshotHeader.cpp
  RadioProjectionDataHeader.cpp
  RadioProjectionHeader.cpp
  RadioSnapshotBuilderHeader.cpp
  RadioStatisticsHeader.cpp
  RadioWaveformHeader.cpp
  RadioWaveformsHeader.cpp)
target_link_libraries (testRadioProjection PRIVATE CORSIKA8RadioProjection
  CORSIKA8RadioSnapshot CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testRadioProjection PRIVATE cxx_std_17)
set_source_files_properties (RadioFixture.cpp RadioSnapshotBuilderHeader.cpp PROPERTIES LANGUAGE CXX)
if (CMAKE_CUDA_COMPILER_LOADED)
  set_source_files_properties (testRadioProjection.cpp RadioProjectionHeader.cpp
    RadioProjectionDataHeader.cpp RadioAtomicOperationsHeader.cpp PROPERTIES LANGUAGE CUDA)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testRadioProjection PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testRadioProjection PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testRadioProjection COMMAND testRadioProjection
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testRadioProjection.xml -r junit)
set_tests_properties (testRadioProjection PROPERTIES LABELS "kokkos;radio"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_subdirectory (accumulator)

add_subdirectory (host)
add_subdirectory (firn)

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testRadioProjection
  testRadioProjection.cpp
  RadioProjectionHeader.cpp
  RadioProjectionDataHeader.cpp
  RadioAtomicOperationsHeader.cpp
)
