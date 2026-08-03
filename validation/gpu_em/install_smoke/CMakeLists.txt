cmake_minimum_required(VERSION 3.24)

project(Corsika8GpuEmInstallSmoke LANGUAGES CXX)

find_package(corsika CONFIG REQUIRED)

add_executable(corsika8_gpu_em_install_smoke main.cpp)
target_link_libraries(
  corsika8_gpu_em_install_smoke
  PRIVATE
    CORSIKA8::CORSIKA8GpuEm
)
