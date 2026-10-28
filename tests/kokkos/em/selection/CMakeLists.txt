# (c) Copyright 2026 CORSIKA Project. Distributed under the 3-clause BSD license.
function (corsika_add_selection_test TARGET)
  add_executable (${TARGET} ${ARGN})
  target_link_libraries (${TARGET} PRIVATE CORSIKA8EmSelection
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
  set_tests_properties (${TARGET} PROPERTIES LABELS "kokkos;selection"
    TIMEOUT 600 RUN_SERIAL TRUE PROCESSORS 2
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false;CORSIKA_NATIVE_TABLE_CACHE=${PROJECT_BINARY_DIR}/native-table-test-cache")
endfunction ()
corsika_add_selection_test (testSelectionContracts testSelectionContracts.cpp
  InteractionPhysicsViewHeader.cpp InteractionSelectionHeader.cpp
  LeptonVertexSelectionHeader.cpp NativeMuonPairCutHeader.cpp)
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  corsika_add_selection_test (testInteractionSelection testInteractionSelection.cpp)
  target_link_libraries (testInteractionSelection PRIVATE NativeTableTestFixture)
endif ()

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testInteractionSelection testInteractionSelection.cpp)

corsika_kokkos_device_test (testSelectionContracts
  testSelectionContracts.cpp
)
