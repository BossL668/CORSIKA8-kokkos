# Production scan/range primitives; no host PROPOSAL dependency.
add_executable (testScanPrimitives testScanPrimitives.cpp
  CompositeExclusiveScanHeader.cpp RangePolicyHeader.cpp)
target_link_libraries (testScanPrimitives PRIVATE CORSIKA8KokkosScan
  CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testScanPrimitives PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testScanPrimitives PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testScanPrimitives PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testScanPrimitives COMMAND testScanPrimitives
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testScanPrimitives.xml -r junit)
set_tests_properties (testScanPrimitives PROPERTIES LABELS "kokkos;runtime;scan"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# Execution waits retain the existing production stream/event semantics.
add_executable (testExecutionWait testExecutionWait.cpp
  CudaCompletionTicketHeader.cpp ResidentExecutionWaitHeader.cpp)
target_link_libraries (testExecutionWait PRIVATE CORSIKA8KokkosRuntime Catch2::Catch2WithMain)
target_compile_features (testExecutionWait PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testExecutionWait PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testExecutionWait PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testExecutionWait COMMAND testExecutionWait
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testExecutionWait.xml -r junit)
set_tests_properties (testExecutionWait PROPERTIES LABELS "kokkos;runtime;wait"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# Production queue ownership and source-ordered particle movement.
add_executable (testParticleQueues testParticleQueues.cpp
  RawViewHeader.cpp GeometricCapacityHeader.cpp ParticleSoAHeader.cpp
  PendingParticleQueueHeader.cpp WavefrontQueueHeader.cpp WavefrontExecutionGuardHeader.cpp)
target_link_libraries (testParticleQueues PRIVATE CORSIKA8ParticleQueues Catch2::Catch2WithMain)
target_compile_features (testParticleQueues PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testParticleQueues PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testParticleQueues PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testParticleQueues COMMAND testParticleQueues
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testParticleQueues.xml -r junit)
set_tests_properties (testParticleQueues PROPERTIES LABELS "kokkos;runtime;queues"
  TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

# Keep generic Runtime tests and add the production ownership/cache contracts.
add_subdirectory (backend)

# CUDA headers also require the device compiler in host reference/header checks.
corsika_kokkos_device_test (testScanPrimitives
  testScanPrimitives.cpp
  CompositeExclusiveScanHeader.cpp
  RangePolicyHeader.cpp
)
corsika_kokkos_device_test (testExecutionWait
  testExecutionWait.cpp
  CudaCompletionTicketHeader.cpp
  ResidentExecutionWaitHeader.cpp
)
corsika_kokkos_device_test (testParticleQueues
  testParticleQueues.cpp
  RawViewHeader.cpp
  GeometricCapacityHeader.cpp
  ParticleSoAHeader.cpp
  PendingParticleQueueHeader.cpp
  WavefrontQueueHeader.cpp
  WavefrontExecutionGuardHeader.cpp
)
