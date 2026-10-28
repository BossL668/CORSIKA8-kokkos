#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

function (corsika_add_native_table_test TARGET)
  add_executable (${TARGET} ${ARGN})
  target_link_libraries (${TARGET}
    PRIVATE CORSIKA8NativeTableQueries Catch2::Catch2WithMain)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  if (CORSIKA_SANITIZERS_ENABLED)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PRIVATE -fsanitize=address,undefined)
  endif ()
  add_test (NAME ${TARGET} COMMAND ${TARGET}
    -o ${PROJECT_BINARY_DIR}/test_outputs/junit-${TARGET}.xml -r junit)
  set_tests_properties (${TARGET} PROPERTIES
    LABELS "kokkos;native_tables"
    TIMEOUT 600
    RUN_SERIAL TRUE
    PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache")
endfunction ()

# These contracts need neither an export-enabled PROPOSAL nor generated tables.
corsika_add_native_table_test (testNativeTableContracts testNativeTableContracts.cpp
  ProposalNativeDataHeader.cpp ProposalNativeQueriesHeader.cpp NativeTableStorageHeader.cpp
  ProposalNativeTableHeader.cpp NativeTableExporterHeader.cpp Sha256Header.cpp)

if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  # This translation unit alone constructs native C8/PROPOSAL objects. Device
  # query translation units see only the owning numerical table declaration.
  add_library (NativeTableTestFixture STATIC NativeTableFixture.cpp
    ProposalNativeExportReference.cpp)
  target_link_libraries (NativeTableTestFixture
    PUBLIC CORSIKA8ProposalBridge PRIVATE CORSIKA8)
  if (NOT CORSIKA_NATIVE_FOCUSED_HARNESS)
    add_library (NativeCalculatorModuleInstantiation OBJECT
      NativeCalculatorModuleInstantiation.cpp)
    target_link_libraries (NativeCalculatorModuleInstantiation PRIVATE CORSIKA8)
  endif ()

  corsika_add_native_table_test (testProposalNativeExport testProposalNativeExport.cpp
    NativeCalculatorViewHeader.cpp)
  target_link_libraries (testProposalNativeExport PRIVATE NativeTableTestFixture CORSIKA8)

  corsika_add_native_table_test (testProposalNativeQueries testProposalNativeQueries.cpp)
  target_link_libraries (testProposalNativeQueries PRIVATE NativeTableTestFixture)

  corsika_add_native_table_test (testProposalNativeQueriesDevice testProposalNativeQueriesDevice.cpp)
  corsika_add_native_table_test (testNativeTableStorage testNativeTableStorage.cpp)
  foreach (TARGET testProposalNativeQueriesDevice testNativeTableStorage)
    target_link_libraries (${TARGET} PRIVATE NativeTableTestFixture CORSIKA8KokkosRuntime)
    if (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "CUDA")
      target_compile_definitions (${TARGET} PRIVATE CORSIKA_EXPECT_CUDA)
    elseif (CORSIKA_DATA_EXPECT_BACKEND STREQUAL "OPENMP")
      target_compile_definitions (${TARGET} PRIVATE CORSIKA_EXPECT_OPENMP)
    endif ()
  endforeach ()
endif ()

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testProposalNativeQueriesDevice
  testProposalNativeQueriesDevice.cpp
)
corsika_kokkos_device_test (testNativeTableStorage
  testNativeTableStorage.cpp
  NativeTableStorageHeader.cpp
)

corsika_kokkos_device_test (testProposalNativeQueries
  testProposalNativeQueries.cpp
)
