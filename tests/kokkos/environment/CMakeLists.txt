#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

CORSIKA_ADD_TEST (testKokkosEnvironmentTopology
  SOURCES testEnvironmentTopology.cpp EnvironmentTopologyHeader.cpp
)
target_link_libraries (testKokkosEnvironmentTopology
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosParticleConversion
)
set_tests_properties (testKokkosEnvironmentTopology PROPERTIES
  LABELS "kokkos;environment;topology"
  TIMEOUT 60
)

add_executable (testKokkosEnvironmentTransfer
  testEnvironmentTransfer.cpp EnvironmentNodeHeader.cpp
)
target_link_libraries (testKokkosEnvironmentTransfer
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosData Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosEnvironmentTransfer PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosEnvironmentTransfer PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosEnvironmentTransfer PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosEnvironmentTransfer
  COMMAND testKokkosEnvironmentTransfer
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosEnvironmentTransfer.xml -r junit
)
set_tests_properties (testKokkosEnvironmentTransfer PROPERTIES
  LABELS "kokkos;environment;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosShapeConversion
  SOURCES testShapeConversion.cpp ShapeConversionHeader.cpp
)
target_link_libraries (testKokkosShapeConversion
  PRIVATE CORSIKA8KokkosEnvironment
)
set_tests_properties (testKokkosShapeConversion PROPERTIES
  LABELS "kokkos;environment;conversion"
  TIMEOUT 60
)

add_executable (testKokkosShapeTransfer
  testShapeTransfer.cpp EnvironmentShapeHeader.cpp
)
target_link_libraries (testKokkosShapeTransfer
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosData Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosShapeTransfer PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosShapeTransfer PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosShapeTransfer PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosShapeTransfer
  COMMAND testKokkosShapeTransfer
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosShapeTransfer.xml -r junit
)
set_tests_properties (testKokkosShapeTransfer PROPERTIES
  LABELS "kokkos;environment;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosModelConversion
  SOURCES testModelConversion.cpp ModelConversionHeader.cpp
)
target_link_libraries (testKokkosModelConversion
  PRIVATE CORSIKA8KokkosEnvironment
)
set_tests_properties (testKokkosModelConversion PROPERTIES
  LABELS "kokkos;environment;conversion"
  TIMEOUT 60
)

add_executable (testKokkosModelTransfer
  testModelTransfer.cpp EnvironmentModelHeader.cpp EnvironmentComponentHeader.cpp
)
target_link_libraries (testKokkosModelTransfer
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosData Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosModelTransfer PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosModelTransfer PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosModelTransfer PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosModelTransfer
  COMMAND testKokkosModelTransfer
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosModelTransfer.xml -r junit
)
set_tests_properties (testKokkosModelTransfer PROPERTIES
  LABELS "kokkos;environment;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosEnvironmentSnapshot
  SOURCES testEnvironmentSnapshot.cpp EnvironmentSnapshotHeader.cpp
)
target_link_libraries (testKokkosEnvironmentSnapshot
  PRIVATE CORSIKA8KokkosEnvironment
)
set_tests_properties (testKokkosEnvironmentSnapshot PROPERTIES
  LABELS "kokkos;environment;snapshot"
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosModelEvaluation
  SOURCES testModelEvaluation.cpp ModelEvaluationHeader.cpp
)
target_link_libraries (testKokkosModelEvaluation
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosModelEvaluation
)
set_tests_properties (testKokkosModelEvaluation PROPERTIES
  LABELS "kokkos;environment;evaluation"
  TIMEOUT 60
)

add_executable (testKokkosModelEvaluationDevice testModelEvaluationDevice.cpp)
target_link_libraries (testKokkosModelEvaluationDevice
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosModelEvaluation Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosModelEvaluationDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosModelEvaluationDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosModelEvaluationDevice PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosModelEvaluationDevice
  COMMAND testKokkosModelEvaluationDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosModelEvaluationDevice.xml -r junit
)
set_tests_properties (testKokkosModelEvaluationDevice PROPERTIES
  LABELS "kokkos;environment;evaluation;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosModelGrammage
  SOURCES testModelGrammage.cpp ModelGrammageHeader.cpp
)
target_link_libraries (testKokkosModelGrammage
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosModelEvaluation
)
set_tests_properties (testKokkosModelGrammage PROPERTIES
  LABELS "kokkos;environment;grammage"
  TIMEOUT 60
)

add_executable (testKokkosModelGrammageDevice testModelGrammageDevice.cpp)
target_link_libraries (testKokkosModelGrammageDevice
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosModelEvaluation Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosModelGrammageDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosModelGrammageDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosModelGrammageDevice PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosModelGrammageDevice
  COMMAND testKokkosModelGrammageDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosModelGrammageDevice.xml -r junit
)
set_tests_properties (testKokkosModelGrammageDevice PROPERTIES
  LABELS "kokkos;environment;grammage;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosEnvironmentSearch
  SOURCES testEnvironmentSearch.cpp EnvironmentSearchHeader.cpp HostEnvironmentTree.cpp
)
target_link_libraries (testKokkosEnvironmentSearch
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime
)
set_tests_properties (testKokkosEnvironmentSearch PROPERTIES
  LABELS "kokkos;environment;search"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

add_executable (testKokkosEnvironmentSearchDevice testEnvironmentSearchDevice.cpp)
target_link_libraries (testKokkosEnvironmentSearchDevice
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosEnvironmentSearch Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosEnvironmentSearchDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosEnvironmentSearchDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosEnvironmentSearchDevice PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosEnvironmentSearchDevice
  COMMAND testKokkosEnvironmentSearchDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosEnvironmentSearchDevice.xml -r junit
)
set_tests_properties (testKokkosEnvironmentSearchDevice PROPERTIES
  LABELS "kokkos;environment;search;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

CORSIKA_ADD_TEST (testKokkosStraightBoundary
  SOURCES testStraightBoundary.cpp StraightBoundaryHeader.cpp HostEnvironmentTree.cpp
)
target_link_libraries (testKokkosStraightBoundary
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime
)
set_tests_properties (testKokkosStraightBoundary PROPERTIES
  LABELS "kokkos;environment;boundary"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

add_executable (testKokkosStraightBoundaryDevice testStraightBoundaryDevice.cpp)
target_link_libraries (testKokkosStraightBoundaryDevice
  PRIVATE CORSIKA8KokkosRuntime CORSIKA8KokkosEnvironmentSearch Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosStraightBoundaryDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosStraightBoundaryDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosStraightBoundaryDevice PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  )
endif ()
add_test (NAME testKokkosStraightBoundaryDevice
  COMMAND testKokkosStraightBoundaryDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosStraightBoundaryDevice.xml -r junit
)
set_tests_properties (testKokkosStraightBoundaryDevice PROPERTIES
  LABELS "kokkos;environment;boundary;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2
  RUN_SERIAL TRUE
  TIMEOUT 60
)

# Curved/straight adapters are constructed independently inside device kernels.
add_executable (testKokkosCurvedBoundary testCurvedBoundary.cpp CurvedBoundaryHeader.cpp)
target_link_libraries (testKokkosCurvedBoundary
  PRIVATE CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
add_executable (testKokkosCurvedBoundaryDevice testCurvedBoundaryDevice.cpp)
target_link_libraries (testKokkosCurvedBoundaryDevice
  PRIVATE CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosCurvedBoundaryDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosCurvedBoundaryDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TEST_TARGET IN ITEMS testKokkosCurvedBoundary testKokkosCurvedBoundaryDevice)
    target_compile_options (${TEST_TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
  endforeach ()
endif ()
add_test (NAME testKokkosCurvedBoundary
  COMMAND testKokkosCurvedBoundary
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosCurvedBoundary.xml -r junit
)
add_test (NAME testKokkosCurvedBoundaryDevice
  COMMAND testKokkosCurvedBoundaryDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosCurvedBoundaryDevice.xml -r junit
)
set_tests_properties (testKokkosCurvedBoundary PROPERTIES
  LABELS "kokkos;environment;curved"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)
set_tests_properties (testKokkosCurvedBoundaryDevice PROPERTIES
  LABELS "kokkos;environment;curved;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)

# Particle-boundary adaptation and the independent observation plane.
add_executable (testKokkosLeptonBoundary
  testLeptonBoundary.cpp LeptonBoundaryHeader.cpp MagneticStepHeader.cpp
  ScalarPhysicalConstantsHeader.cpp
)
target_link_libraries (testKokkosLeptonBoundary
  PRIVATE CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
add_executable (testKokkosLeptonBoundaryDevice testLeptonBoundaryDevice.cpp)
target_link_libraries (testKokkosLeptonBoundaryDevice
  PRIVATE CORSIKA8KokkosEnvironmentSearch CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosLeptonBoundaryDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosLeptonBoundaryDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TEST_TARGET IN ITEMS testKokkosLeptonBoundary testKokkosLeptonBoundaryDevice)
    target_compile_options (${TEST_TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
  endforeach ()
endif ()
add_test (NAME testKokkosLeptonBoundary
  COMMAND testKokkosLeptonBoundary
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosLeptonBoundary.xml -r junit
)
add_test (NAME testKokkosLeptonBoundaryDevice
  COMMAND testKokkosLeptonBoundaryDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosLeptonBoundaryDevice.xml -r junit
)
set_tests_properties (testKokkosLeptonBoundary PROPERTIES
  LABELS "kokkos;environment;lepton-boundary"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)
set_tests_properties (testKokkosLeptonBoundaryDevice PROPERTIES
  LABELS "kokkos;environment;lepton-boundary;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)

# Uniform-field magnetic step limits and particle advance.
add_executable (testKokkosUniformMagneticField
  testUniformMagneticField.cpp UniformMagneticFieldHeader.cpp
)
target_link_libraries (testKokkosUniformMagneticField
  PRIVATE CORSIKA8KokkosData CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
add_executable (testKokkosUniformMagneticFieldDevice testUniformMagneticFieldDevice.cpp)
target_link_libraries (testKokkosUniformMagneticFieldDevice
  PRIVATE CORSIKA8KokkosData CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosUniformMagneticFieldDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosUniformMagneticFieldDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TEST_TARGET IN ITEMS testKokkosUniformMagneticField testKokkosUniformMagneticFieldDevice)
    target_compile_options (${TEST_TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
  endforeach ()
endif ()
add_test (NAME testKokkosUniformMagneticField
  COMMAND testKokkosUniformMagneticField
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosUniformMagneticField.xml -r junit
)
add_test (NAME testKokkosUniformMagneticFieldDevice
  COMMAND testKokkosUniformMagneticFieldDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosUniformMagneticFieldDevice.xml -r junit
)
set_tests_properties (testKokkosUniformMagneticField PROPERTIES
  LABELS "kokkos;environment;uniform-magnetic-field"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)
set_tests_properties (testKokkosUniformMagneticFieldDevice PROPERTIES
  LABELS "kokkos;environment;uniform-magnetic-field;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)

# NodeTree environment queries and straight observation plane.
add_executable (testKokkosEnvironmentQuery
  testEnvironmentQuery.cpp EnvironmentQueryAtmosphere.cpp
  EnvironmentQueryHeader.cpp EnvironmentViewHeader.cpp ObservationPlaneHeader.cpp
)
target_link_libraries (testKokkosEnvironmentQuery
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
add_executable (testKokkosEnvironmentQueryDevice
  testEnvironmentQueryDevice.cpp EnvironmentQueryAtmosphere.cpp
)
target_link_libraries (testKokkosEnvironmentQueryDevice
  PRIVATE CORSIKA8KokkosEnvironment CORSIKA8KokkosRuntime Catch2::Catch2WithMain
)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
  target_compile_definitions (testKokkosEnvironmentQueryDevice PRIVATE CORSIKA_EXPECT_CUDA)
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
  target_compile_definitions (testKokkosEnvironmentQueryDevice PRIVATE CORSIKA_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TEST_TARGET IN ITEMS testKokkosEnvironmentQuery testKokkosEnvironmentQueryDevice)
    target_compile_options (${TEST_TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
  endforeach ()
endif ()
add_test (NAME testKokkosEnvironmentQuery
  COMMAND testKokkosEnvironmentQuery
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosEnvironmentQuery.xml -r junit
)
add_test (NAME testKokkosEnvironmentQueryDevice
  COMMAND testKokkosEnvironmentQueryDevice
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosEnvironmentQueryDevice.xml -r junit
)
set_tests_properties (testKokkosEnvironmentQuery PROPERTIES
  LABELS "kokkos;environment;query"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)
set_tests_properties (testKokkosEnvironmentQueryDevice PROPERTIES
  LABELS "kokkos;environment;query;execution"
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false"
  PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
)


# Device array ownership, host snapshot binding, and shared borrowed queries.
add_executable (testKokkosEnvironmentStorage testEnvironmentStorage.cpp
  EnvironmentStorageHeader.cpp EnvironmentDataViewHeader.cpp EnvironmentBorrowedViewHeader.cpp)
target_link_libraries (testKokkosEnvironmentStorage PRIVATE CORSIKA8EnvironmentStorage
  Catch2::Catch2WithMain)
add_executable (testKokkosEnvironmentSnapshotData testEnvironmentSnapshotData.cpp
  EnvironmentSnapshotDataHeader.cpp)
target_link_libraries (testKokkosEnvironmentSnapshotData PRIVATE CORSIKA8KokkosEnvironment
  Catch2::Catch2WithMain)
add_executable (testKokkosEnvironmentBorrowedQuery testEnvironmentBorrowedQuery.cpp
  EnvironmentQueryAtmosphere.cpp)
target_link_libraries (testKokkosEnvironmentBorrowedQuery PRIVATE CORSIKA8EnvironmentStorage
  CORSIKA8KokkosEnvironment Catch2::Catch2WithMain)
foreach (TEST_TARGET IN ITEMS testKokkosEnvironmentStorage testKokkosEnvironmentSnapshotData
    testKokkosEnvironmentBorrowedQuery)
  target_compile_features (${TEST_TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TEST_TARGET} PRIVATE -O1 -g -fno-omit-frame-pointer
      -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TEST_TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TEST_TARGET} COMMAND ${TEST_TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TEST_TARGET}.xml -r junit)
  set_tests_properties (${TEST_TARGET} PROPERTIES LABELS "kokkos;environment;storage"
    TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach ()

# Independent firn profiles: native C8 host objects and device numerical queries.
add_executable (testKokkosFirnEnvironment
  testFirnEnvironment.cpp FirnProbe.cpp FirnModelHeader.cpp
  ExponentialFirnHeader.cpp ExponentialFirnProfileHeader.cpp)
target_link_libraries (testKokkosFirnEnvironment PRIVATE
  CORSIKA8EnvironmentStorage CORSIKA8KokkosEnvironment Catch2::Catch2WithMain)
target_compile_features (testKokkosFirnEnvironment PRIVATE cxx_std_17)
if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA" OR Kokkos_DEVICES MATCHES "CUDA")
  target_compile_definitions (testKokkosFirnEnvironment PRIVATE FIRN_EXPECT_CUDA)
  if (CMAKE_CUDA_COMPILER_LOADED)
    set_source_files_properties (FirnProbe.cpp FirnModelHeader.cpp
      PROPERTIES LANGUAGE CUDA)
  endif ()
elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP" OR Kokkos_DEVICES MATCHES "OPENMP")
  target_compile_definitions (testKokkosFirnEnvironment PRIVATE FIRN_EXPECT_OPENMP)
endif ()
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testKokkosFirnEnvironment PRIVATE -O1 -g
    -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testKokkosFirnEnvironment PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testKokkosFirnEnvironment COMMAND testKokkosFirnEnvironment
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testKokkosFirnEnvironment.xml -r junit)
set_tests_properties (testKokkosFirnEnvironment PROPERTIES
  LABELS "kokkos;environment;firn" TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testKokkosEnvironmentTransfer testEnvironmentTransfer.cpp)
corsika_kokkos_device_test (testKokkosShapeTransfer testShapeTransfer.cpp)
corsika_kokkos_device_test (testKokkosModelTransfer testModelTransfer.cpp)
corsika_kokkos_device_test (testKokkosModelEvaluationDevice testModelEvaluationDevice.cpp)
corsika_kokkos_device_test (testKokkosModelGrammageDevice testModelGrammageDevice.cpp)
corsika_kokkos_device_test (testKokkosEnvironmentSearchDevice
  testEnvironmentSearchDevice.cpp
)
corsika_kokkos_device_test (testKokkosStraightBoundaryDevice
  testStraightBoundaryDevice.cpp
)
corsika_kokkos_device_test (testKokkosCurvedBoundaryDevice testCurvedBoundaryDevice.cpp)
corsika_kokkos_device_test (testKokkosLeptonBoundaryDevice testLeptonBoundaryDevice.cpp)
corsika_kokkos_device_test (testKokkosUniformMagneticFieldDevice
  testUniformMagneticFieldDevice.cpp
)
corsika_kokkos_device_test (testKokkosEnvironmentQueryDevice
  testEnvironmentQueryDevice.cpp
)
corsika_kokkos_device_test (testKokkosEnvironmentStorage
  testEnvironmentStorage.cpp
  EnvironmentStorageHeader.cpp
  EnvironmentDataViewHeader.cpp
  EnvironmentBorrowedViewHeader.cpp
)
corsika_kokkos_device_test (testKokkosEnvironmentBorrowedQuery
  testEnvironmentBorrowedQuery.cpp
)
corsika_kokkos_device_test (testKokkosFirnEnvironment FirnProbe.cpp FirnModelHeader.cpp)



corsika_kokkos_device_test (testKokkosEnvironmentSearch
  HostEnvironmentTree.cpp
  EnvironmentSearchHeader.cpp
)

corsika_kokkos_device_test (testKokkosStraightBoundary
  HostEnvironmentTree.cpp
  StraightBoundaryHeader.cpp
)

corsika_kokkos_device_test (testKokkosCurvedBoundary
  testCurvedBoundary.cpp
  CurvedBoundaryHeader.cpp
)

corsika_kokkos_device_test (testKokkosLeptonBoundary
  testLeptonBoundary.cpp
  LeptonBoundaryHeader.cpp
)

corsika_kokkos_device_test (testKokkosUniformMagneticField
  testUniformMagneticField.cpp
  UniformMagneticFieldHeader.cpp
)

corsika_kokkos_device_test (testKokkosEnvironmentQuery
  testEnvironmentQuery.cpp
  EnvironmentViewHeader.cpp
)
