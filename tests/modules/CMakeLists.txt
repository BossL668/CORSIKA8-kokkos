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
    add_subdirectory (fluka)
endif()

CORSIKA_ADD_TEST (testModules SOURCES ${test_modules_sources})

target_compile_definitions (
  testModules
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

CORSIKA_ADD_TEST (testRadioTrackCapture SOURCES testRadioTrackCapture.cpp
  RadioTrackSegmentHeader.cpp RadioTrackSinkHeader.cpp)

CORSIKA_ADD_TEST(testRadioOutputLifecycle SOURCES radio_output/testRadioOutputLifecycle.cpp)

CORSIKA_ADD_TEST (testSpecifiedProposal SOURCES proposal/testSpecifiedInteraction.cpp
  proposal/ProposalInteractionRecordHeader.cpp proposal/ProposalRateProviderHeader.cpp
  proposal/ProposalFinalStateGeneratorHeader.cpp)
file (MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/specified-proposal-data/PROPOSAL)
set_tests_properties (testSpecifiedProposal PROPERTIES LABELS "proposal;fallback"
  RUN_SERIAL TRUE TIMEOUT 1200
  ENVIRONMENT "CORSIKA_DATA=${PROJECT_BINARY_DIR}/specified-proposal-data;OMP_NUM_THREADS=1")

add_subdirectory (native_cut)

add_subdirectory (photohadronic)
