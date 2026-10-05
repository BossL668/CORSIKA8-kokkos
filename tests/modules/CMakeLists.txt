if (CORSIKA_ENABLE_EGS4 AND CORSIKA_EGS4_BUILD_TESTS)
  add_subdirectory (egs4)
endif ()

set (test_modules_sources
  TestMain.cpp
  testStackInspector.cpp
  testTracking.cpp
  ## testExecTime.cpp --> needs to be fixed, see #326
  testObservationPlane.cpp
  testObservationVolume.cpp
  testQGSJetII.cpp # temp disable
  testQGSJetIII.cpp
  testProposal.cpp
  testPythia8.cpp
  testTAUOLA.cpp
  testUrQMD.cpp
  #  testCONEX.cpp TODO re-enable
  testParticleCut.cpp
  testProductionProfile.cpp
  testSibyll.cpp
  testEpos.cpp
  testRadio.cpp
  testEMThinning.cpp
  testSophia.cpp
  #testRandom.cpp # temp disable for qgs2 test
)

if (NOT "$CACHE{C8_FLUKA_ENABLED}" STREQUAL "")
    list(APPEND test_modules_sources "testFluka.cpp")
    message("testFluka enabled")
    find_package(Threads REQUIRED)
    add_executable(testFlukaDiagnosticTimer testFlukaDiagnosticTimer.cpp
        ../../src/modules/fluka/FlukaDiagnosticTimer.cpp)
    target_link_libraries(testFlukaDiagnosticTimer PRIVATE Threads::Threads)
    add_test(NAME testFlukaDiagnosticTimer COMMAND testFlukaDiagnosticTimer)
    set_tests_properties(testFlukaDiagnosticTimer PROPERTIES TIMEOUT 20)
endif()

CORSIKA_ADD_TEST (testModules SOURCES ${test_modules_sources})
CORSIKA_ADD_TEST (testStoppedPositron SOURCES testStoppedPositron.cpp)

# Independent finite-mountain neutrino vertex/rate oracle. Keeping this separate
# avoids running the complete hadronic module suite for a geometry integration.
CORSIKA_ADD_TEST (testMountainNeutrino SOURCES testMountainNeutrino.cpp)
CORSIKA_ADD_TEST (testMountainNeutrinoExtended SOURCES testMountainNeutrinoExtended.cpp)
CORSIKA_ADD_TEST (testMountainOriginalLeptons SOURCES testMountainOriginalLeptons.cpp)
CORSIKA_ADD_TEST (testMountainTauSpin SOURCES testMountainTauSpin.cpp)
CORSIKA_ADD_TEST (testMountainRadio SOURCES testMountainRadio.cpp)
CORSIKA_ADD_TEST (testTerrainEnvironment SOURCES testTerrainEnvironment.cpp)
CORSIKA_ADD_TEST (testMountainMaterials SOURCES testMountainMaterials.cpp)
target_compile_definitions(testMountainMaterials PRIVATE MATERIAL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
if(TARGET C8TerrainFlukaMaterialBridge)
  add_executable(testFlukaMaterialOracle EXCLUDE_FROM_ALL
    ../../validation/terrain/material_models/FlukaMaterialOracle.cpp
    $<TARGET_OBJECTS:C8TerrainFlukaMaterialBridge>)
  target_link_libraries(testFlukaMaterialOracle PRIVATE CORSIKA8)
endif()

if(CORSIKA_BUILD_MOUNTAIN_APPLICATION AND CORSIKA_ENABLE_KOKKOS AND WITH_FLUKA)
  add_executable(testMaterialOracle EXCLUDE_FROM_ALL ../../validation/terrain/material_models/MaterialOracle.cpp)
  target_link_libraries(testMaterialOracle PRIVATE CORSIKA8 CORSIKA8TerrainEm)
  add_dependencies(testMaterialOracle GenParticlesHeaders GenMediaProperties)
  add_executable(testInterfaceTransport EXCLUDE_FROM_ALL testInterfaceTransport.cpp)
  target_link_libraries(testInterfaceTransport PRIVATE CORSIKA8 CORSIKA8TerrainEm)
  add_dependencies(testInterfaceTransport GenParticlesHeaders GenMediaProperties)
  foreach(variant ordinary same tetra field swapped fieldonly)
    add_test(NAME InterfaceTransport_${variant} COMMAND testInterfaceTransport ${variant})
    set_tests_properties(InterfaceTransport_${variant} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=${C8_INTERFACE_TEST_THREADS};OMP_PROC_BIND=spread;OMP_PLACES=threads;C8_INTERFACE_TEST_THREADS=${C8_INTERFACE_TEST_THREADS}"
      RESOURCE_LOCK interface_proposal_cache TIMEOUT 1800)
  endforeach()
endif()

target_compile_definitions (
  testModules
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
