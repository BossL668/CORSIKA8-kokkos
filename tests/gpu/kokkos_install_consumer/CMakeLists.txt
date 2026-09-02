cmake_minimum_required(VERSION 3.24)

project(c8_kokkos_install_consumer LANGUAGES CXX)

find_package(corsika CONFIG REQUIRED)

add_executable(c8_kokkos_install_consumer main.cpp)
target_link_libraries(
  c8_kokkos_install_consumer PRIVATE CORSIKA8::CORSIKA8KokkosEm)
