# Host capture plus bounded replay into the same production accumulator.
add_executable (testHostRadio
  testHostRadioCollector.cpp testRadioTrackSpool.cpp testRadioSpoolCompatibility.cpp
  testRadioSinkScope.cpp testRadioSpoolReplay.cpp
  HostRadioRecordHeader.cpp
  HostRadioCollectorHeader.cpp
  ScopedRadioTrackSinkHeader.cpp
  RadioTrackSpoolFormatHeader.cpp
  RadioTrackSpoolReaderHeader.cpp
  RadioTrackSpoolWriterHeader.cpp
)
target_link_libraries (testHostRadio PRIVATE CORSIKA8HostRadio Catch2::Catch2WithMain)
target_compile_features (testHostRadio PRIVATE cxx_std_17)
if (CMAKE_CUDA_COMPILER_LOADED)
  set_source_files_properties (testRadioSpoolReplay.cpp PROPERTIES LANGUAGE CUDA)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testHostRadio PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testHostRadio PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testHostRadio COMMAND testHostRadio
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testHostRadio.xml -r junit)
set_tests_properties (testHostRadio PROPERTIES LABELS "kokkos;radio;host"
  TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testRadioSpoolReplay testRadioSpoolReplay.cpp)
