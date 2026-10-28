set (test_stack_sources
  TestMain.cpp
  testHistoryStack.cpp
  testGeometryNodeStackExtension.cpp
  testWeightStackExtension.cpp
  testDummyStack.cpp
  testVectorStack.cpp
  )

CORSIKA_ADD_TEST (testStack SOURCES ${test_stack_sources})

CORSIKA_ADD_TEST (testTransportIdentity SOURCES testTransportIdentity.cpp
  TransportIdentityStackExtensionHeader.cpp)
set_tests_properties (testTransportIdentity PROPERTIES LABELS "stack;transport" TIMEOUT 60)
