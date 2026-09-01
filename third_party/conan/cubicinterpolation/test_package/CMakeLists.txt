cmake_minimum_required(VERSION 3.15)
project(cubicinterpolation_export_test LANGUAGES CXX)

find_package(CubicInterpolation CONFIG REQUIRED)

option(C8GPU_BUILD_EXPORT_TEST "Build the patched export-API test" ON)
if(C8GPU_BUILD_EXPORT_TEST)
  add_executable(cubic_export_test test_package.cpp)
  target_compile_features(cubic_export_test PRIVATE cxx_std_17)
  target_link_libraries(cubic_export_test PRIVATE
    CubicInterpolation::CubicInterpolation)
endif()

add_executable(cubic_ordinary_api_test ordinary_api.cpp)
target_compile_features(cubic_ordinary_api_test PRIVATE cxx_std_17)
target_link_libraries(cubic_ordinary_api_test PRIVATE
  CubicInterpolation::CubicInterpolation)
