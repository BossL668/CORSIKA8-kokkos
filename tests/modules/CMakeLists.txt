set (test_modules_sources
  TestMain.cpp
  testStackInspector.cpp
  testTracking.cpp
  ## testExecTime.cpp --> needs to be fixed, see #326
  testObservationPlane.cpp
  testObservationVolume.cpp
  #testQGSJetII.cpp # temp disable
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
endif()

CORSIKA_ADD_TEST (testModules SOURCES ${test_modules_sources})

target_compile_definitions (
  testModules
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
