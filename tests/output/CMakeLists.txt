set (test_output_sources
  testDummyOutputManager.cpp
  testInteractionWriter.cpp
  TestMain.cpp
  testOutputManager.cpp
  testSimulationTiming.cpp
  testShowerSummary.cpp
  testParquetStreamer.cpp
  testWriterObservationPlane.cpp
  testWriterTracks.cpp
  testWriterEnergyLoss.cpp
  testWriterLongitudinal.cpp
  testWriterOff.cpp
  testWritePrimary.cpp
  )

CORSIKA_ADD_TEST (testOutput SOURCES ${test_output_sources})

target_compile_definitions (
  testOutput
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
