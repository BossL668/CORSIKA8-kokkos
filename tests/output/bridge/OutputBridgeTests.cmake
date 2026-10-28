# Prepared integration fragment; not yet added to the receiver.
# Include from tests/output after copying the frozen fixtures to bridge/.
# These are host-only writer/adapter tests and run with Kokkos OFF as well as ON.
add_executable (testOutputBridge bridge/OutputBridge.cpp bridge/Headers.cpp)
target_link_libraries (testOutputBridge PRIVATE CORSIKA8)
target_compile_features (testOutputBridge PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testOutputBridge PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testOutputBridge PRIVATE -fsanitize=address,undefined)
endif ()
foreach (mode IN ITEMS writers sink completion writer-off)
  add_test (NAME "testOutputBridge_${mode}" COMMAND testOutputBridge "${mode}")
  set_tests_properties ("testOutputBridge_${mode}" PROPERTIES
    LABELS "output;production-comparison;native-compatibility"
    TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach ()
