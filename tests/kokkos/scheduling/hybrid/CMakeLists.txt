# Deterministic protocol peer: this test does not replace a physical generator.
find_package (Threads REQUIRED)
add_executable (hybridProtocolTestWorker HybridProtocolTestWorker.cpp)
target_link_libraries (hybridProtocolTestWorker PRIVATE CORSIKA8 Threads::Threads)
target_compile_features (hybridProtocolTestWorker PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (hybridProtocolTestWorker PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (hybridProtocolTestWorker PRIVATE -fsanitize=address,undefined)
endif ()
CORSIKA_ADD_TEST (testHybridCascade SOURCES testHybridCascade.cpp HybridCascadeHeader.cpp)
target_include_directories (testHybridCascade PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/oracle" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries (testHybridCascade PRIVATE Threads::Threads)
add_dependencies (testHybridCascade hybridProtocolTestWorker)
set_tests_properties (testHybridCascade PROPERTIES
  LABELS "kokkos;scheduling;hybrid;host" RUN_SERIAL TRUE TIMEOUT 180
  ENVIRONMENT "HYBRID_TEST_WORKER=$<TARGET_FILE:hybridProtocolTestWorker>")
