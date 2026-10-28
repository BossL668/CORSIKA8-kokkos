# Explicit counter policies; native defaults are tested in tests/framework.
CORSIKA_ADD_TEST (testInteractionDeferralCounter
  SOURCES testInteractionDeferralCounter.cpp CounterHeaderFirst.cpp AdapterHeaderFirst.cpp
    NativeCounterOtherTu.cpp DeferredCounterOtherTu.cpp)
CORSIKA_ADD_TEST (testDeferredCounterCommit SOURCES testDeferredCounterCommit.cpp)
foreach (TARGET testInteractionDeferralCounter testDeferredCounterCommit)
  target_include_directories (${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/oracle" "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set_tests_properties (${TARGET} PROPERTIES
    LABELS "kokkos;scheduling;deferral;host" RUN_SERIAL TRUE TIMEOUT 180)
endforeach ()
