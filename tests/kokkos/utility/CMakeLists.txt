#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

# Reuse numerical geometry's compilation requirements without physics libraries.
add_executable (testKokkosPolynomialRoots
  testPolynomialRoots.cpp PolynomialRootsHeader.cpp
)
target_link_libraries (testKokkosPolynomialRoots
  PRIVATE CORSIKA8KokkosEnvironmentSearch Catch2::Catch2WithMain
)

add_executable (testKokkosPolynomialRootsDevice testPolynomialRootsDevice.cpp)
target_link_libraries (testKokkosPolynomialRootsDevice
  PRIVATE CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosPolynomialRootsDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosPolynomialRootsDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()

# The host-only test does not inherit Runtime's sanitizer link flags.
if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TEST_TARGET IN ITEMS testKokkosPolynomialRoots testKokkosPolynomialRootsDevice)
    target_compile_options (${TEST_TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
  endforeach ()
  target_link_options (testKokkosPolynomialRoots PRIVATE
    -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()

add_test (NAME testKokkosPolynomialRoots
  COMMAND testKokkosPolynomialRoots
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosPolynomialRoots.xml -r junit
)
set_tests_properties (testKokkosPolynomialRoots PROPERTIES
  LABELS "kokkos;utility;roots"
  TIMEOUT 60
)
add_test (NAME testKokkosPolynomialRootsDevice
  COMMAND testKokkosPolynomialRootsDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosPolynomialRootsDevice.xml -r junit
)
set_tests_properties (testKokkosPolynomialRootsDevice PROPERTIES
  LABELS "kokkos;utility;roots;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testKokkosPolynomialRootsDevice testPolynomialRootsDevice.cpp)

corsika_kokkos_device_test (testKokkosPolynomialRoots
  testPolynomialRoots.cpp
  PolynomialRootsHeader.cpp
)
