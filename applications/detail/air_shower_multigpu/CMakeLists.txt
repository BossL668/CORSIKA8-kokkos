# Host-only coordinator: no Kokkos context and no Python runtime dependency.
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
add_library(c8_multigpu_coordinator STATIC NativeCoordinator.cpp NativeIO.cpp)
target_compile_features(c8_multigpu_coordinator PUBLIC cxx_std_17)
if(CORSIKA_ENABLE_EGS4)
  target_compile_definitions(c8_multigpu_coordinator PRIVATE CORSIKA8_WITH_NATIVE_EGS4=1)
endif()
target_include_directories(c8_multigpu_coordinator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(c8_multigpu_coordinator PUBLIC
  nlohmann_json::nlohmann_json Parquet::parquet_static yaml-cpp::yaml-cpp
  CLI11::CLI11 OpenSSL::Crypto)
