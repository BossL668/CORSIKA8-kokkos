cmake_minimum_required(VERSION 3.15)
project(proposal_native_export_test LANGUAGES CXX)

find_package(PROPOSAL CONFIG REQUIRED)

option(C8GPU_BUILD_EXPORT_TEST "Build the patched export-API test" ON)
if(C8GPU_BUILD_EXPORT_TEST)
  add_executable(proposal_export_test test_package.cpp)
  target_compile_features(proposal_export_test PRIVATE cxx_std_17)
  target_link_libraries(proposal_export_test PRIVATE PROPOSAL::PROPOSAL)
endif()

add_executable(proposal_ordinary_api_test ordinary_api.cpp)
target_compile_features(proposal_ordinary_api_test PRIVATE cxx_std_17)
target_link_libraries(proposal_ordinary_api_test PRIVATE PROPOSAL::PROPOSAL)
