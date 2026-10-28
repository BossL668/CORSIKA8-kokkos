#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

add_executable (testKokkosDataContract
  testKokkosDataContract.cpp
  ParticleStateHeader.cpp
  StepRecordHeader.cpp
)
target_link_libraries (testKokkosDataContract
  PRIVATE CORSIKA8KokkosData Catch2::Catch2WithMain
)
target_compile_features (testKokkosDataContract PRIVATE cxx_std_17)

add_executable (testKokkosDataTransfer testKokkosDataTransfer.cpp)
target_link_libraries (testKokkosDataTransfer
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosData Catch2::Catch2WithMain
)
target_compile_features (testKokkosDataTransfer PRIVATE cxx_std_17)

# Explicit audits can require a backend. Ordinary project builds test their
# configured default execution space, including Serial, without an override.
set (CORSIKA_DATA_EXPECT_BACKEND "" CACHE STRING "Expected data-test backend: CUDA or OPENMP")
set_property (CACHE CORSIKA_DATA_EXPECT_BACKEND PROPERTY STRINGS "" CUDA OPENMP)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosDataTransfer PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosDataTransfer PRIVATE CORSIKA_EXPECT_OPENMP)
elseif (NOT CORSIKA_DATA_EXPECT_BACKEND STREQUAL "")
  message (FATAL_ERROR "CORSIKA_DATA_EXPECT_BACKEND must be empty, CUDA, or OPENMP.")
endif ()

# The enclosing runtime configuration rejects sanitizer/device combinations.
# The contract target has no Runtime dependency, so it also needs link flags.
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosDataContract PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
  target_link_options (testKokkosDataContract PRIVATE
    -fsanitize=address,undefined -fno-sanitize-recover=all
  )
  target_compile_options (testKokkosDataTransfer PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()

file (MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_outputs)
foreach (SCENARIO IN ITEMS layout defaults patterns)
  set (TEST_NAME testKokkosDataContract_${SCENARIO})
  add_test (NAME ${TEST_NAME}
    COMMAND testKokkosDataContract "[data_${SCENARIO}]"
      -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TEST_NAME}.xml -r junit
  )
  set_tests_properties (${TEST_NAME} PROPERTIES
    LABELS "kokkos;data;contract"
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
    PROCESSORS 2
    RUN_SERIAL TRUE
    TIMEOUT 60
  )
endforeach ()

# Each case owns a fresh Runtime and must run in its own process.
foreach (SCENARIO IN ITEMS transfer device_defaults)
  set (TEST_NAME testKokkosDataTransfer_${SCENARIO})
  add_test (NAME ${TEST_NAME}
    COMMAND testKokkosDataTransfer "[data_${SCENARIO}]"
      -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TEST_NAME}.xml -r junit
  )
  set_tests_properties (${TEST_NAME} PROPERTIES
    LABELS "kokkos;data;execution"
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
    PROCESSORS 2
    RUN_SERIAL TRUE
    TIMEOUT 60
  )
endforeach ()

# Unlike plain record tests, conversions intentionally use the original C8 API.
CORSIKA_ADD_TEST (testKokkosParticleConversion
  SOURCES testParticleConversion.cpp ParticleConversionHeader.cpp
)
target_link_libraries (testKokkosParticleConversion
  PRIVATE CORSIKA8KokkosParticleConversion
)
set_tests_properties (testKokkosParticleConversion PROPERTIES
  LABELS "kokkos;data;conversion"
  TIMEOUT 60
)

# Compile the actual execution-space queries with the selected device compiler.
corsika_kokkos_device_test (testKokkosDataTransfer testKokkosDataTransfer.cpp)
