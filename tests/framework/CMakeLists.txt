
set (test_framework_sources  
  TestMain.cpp
  testFourVector.cpp
  testSaveBoostHistogram.cpp
  testClassTimer.cpp
  testDynamicInteractionProcess.cpp;
  testLogging.cpp
  testParticles.cpp
  testStackInterface.cpp
  testProcessSequence.cpp
  testCOMBoost.cpp
  # testCorsikaFenv.cpp # does not work because of use of exceptions in catch2
  testFunctionTimer.cpp
  testSecondaryView.cpp
  testGeometry.cpp
  testCombinedStack.cpp
  testUnits.cpp
  testCascade.cpp 
  testRandom.cpp
  testNullModel.cpp  
  testHelix.cpp
  testInteractionCounter.cpp
  testInteractionLengthModifier.cpp
  testSolver.cpp
  testEnergyMomentum.cpp
  )

CORSIKA_ADD_TEST (testFramework SOURCES ${test_framework_sources})

target_compile_definitions (
  testFramework
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

CORSIKA_ADD_TEST (testNextEnvironmentBoundary
  SOURCES geometry/testNextEnvironmentBoundary.cpp
    geometry/NextEnvironmentBoundaryHeader.cpp
)
set_tests_properties (testNextEnvironmentBoundary PROPERTIES
  LABELS "geometry;boundary"
  TIMEOUT 60
)

# The default counter remains available without the optional Kokkos headers.
CORSIKA_ADD_TEST (testNativeCounterDefault
  SOURCES process/testNativeCounterDefault.cpp process/NativeCounterHeader.cpp)
set_tests_properties (testNativeCounterDefault PROPERTIES
  LABELS "process;counter;native" RUN_SERIAL TRUE TIMEOUT 180)
