# (c) Copyright 2026 CORSIKA Project. Distributed under the 3-clause BSD license.
function (corsika_add_stochastic_test TARGET)
  add_executable (${TARGET} ${ARGN})
  target_link_libraries (${TARGET} PRIVATE CORSIKA8EmStochastic
    CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
    target_compile_definitions (${TARGET} PRIVATE CORSIKA_EXPECT_CUDA)
  elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
    target_compile_definitions (${TARGET} PRIVATE CORSIKA_EXPECT_OPENMP)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES LABELS "kokkos;stochastic"
    TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache")
endfunction ()

corsika_add_stochastic_test (testStochasticContracts testStochasticContracts.cpp
  PhiloxHeader.cpp RandomDomainsHeader.cpp ParticleIdHeader.cpp
  ProcessCapabilitiesHeader.cpp TableQueryHeader.cpp FallbackRequestHeader.cpp)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  corsika_add_stochastic_test (testTableDispatch testTableDispatch.cpp)
  target_link_libraries (testTableDispatch PRIVATE NativeTableTestFixture)
endif ()

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testStochasticContracts testStochasticContracts.cpp)
corsika_kokkos_device_test (testTableDispatch testTableDispatch.cpp)
