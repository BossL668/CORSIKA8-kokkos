#
# (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

cmake_minimum_required (VERSION 3.9)

#+++++++++++++++++++++++++++++
# project name
# version is:  "8.major.minor.patch"
#   major: API changes
#   minor: no API changes
#   patch: bug fix and small improvements
#
set (c8_version 8.0.0.0)
project (
  corsika
  VERSION ${c8_version}
  DESCRIPTION "CORSIKA C++ project (alpha status)"
  LANGUAGES CXX
  )

option (
  CORSIKA_ENABLE_CUDA
  "Build the experimental CUDA electromagnetic transport backend"
  OFF
  )

if (CORSIKA_ENABLE_CUDA)
  if (${CMAKE_VERSION} VERSION_LESS "3.24")
    message (
      FATAL_ERROR
      "| CORSIKA8 > CORSIKA_ENABLE_CUDA requires CMake 3.24 or newer."
      )
  endif ()
  enable_language (CUDA)
  set (CMAKE_CUDA_STANDARD 17)
  set (CMAKE_CUDA_STANDARD_REQUIRED ON)
  set (CMAKE_CUDA_EXTENSIONS OFF)
  find_package (CUDAToolkit 12 REQUIRED)
endif ()

#+++++++++++++++++++++++++++++
# for pre-defined standard path
#
include (GNUInstallDirs)

#+++++++++++++++++++++++++++++
# prevent in-source builds and give warning message
#
if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
  message (FATAL_ERROR "In-source builds are disabled.
    Please create a build-dir and use `cmake <source-dir>` inside it.
    NOTE: cmake will now create CMakeCache.txt and CMakeFiles/*.
          You must delete them, or cmake will refuse to work.")
endif ()

#++++++++++++++++++++++++++++
# cmake version-specific settings
#
# https://cmake.org/cmake/help/latest/policy/CMP0079.html
if (POLICY CMP0079)
    cmake_policy (SET CMP0079 NEW)
endif ()
# Download timestamp for external modules
if (POLICY CMP0135)
    cmake_policy (SET CMP0135 OLD)
endif ()
# AUtomatically add extensions to filenames if needed (old behaviour)
if (POLICY CMP0115)
    cmake_policy (SET CMP0115 OLD)
endif ()

#+++++++++++++++++++++++++++++
# warn user if system is not UNIX
#
if (NOT UNIX)
  message (FATAL_ERROR "| CORSIKA8 > This is an unsupported system.")
endif ()

#+++++++++++++++++++++++++++++
# cmake path dir, and cmake config
#
set (CORSIKA8_CMAKE_DIR "${PROJECT_SOURCE_DIR}/cmake")

set (CMAKE_MODULE_PATH "${CORSIKA8_CMAKE_DIR}" ${CMAKE_MODULE_PATH})

if(DEFINED CONAN_CMAKE_DIR)
  list(APPEND CMAKE_MODULE_PATH "${CONAN_CMAKE_DIR}")
endif(DEFINED CONAN_CMAKE_DIR)

set (CMAKE_VERBOSE_MAKEFILE  OFF) # this can be done with `make VERBOSE=1`
# ignore many irrelevant Up-to-date messages during install
set (CMAKE_INSTALL_MESSAGE LAZY)

#+++++++++++++++++++++++++++++
# Extra cmake functions for registering and running tests
#
include (corsikaUtilities) # extra cmake functions

#+++++++++++++++++++++++++++++
# Setup hardware and infrastructure dependent defines, and general settings
#
include (corsikaDefines)

#+++++++++++++++++++++++++++++
# check if compiler is C++17 compliant
#
include (CheckCXXCompilerFlag)
check_CXX_compiler_flag ("-std=c++17" COMPILER_SUPPORTS_CXX17)
if (NOT COMPILER_SUPPORTS_CXX17)
 message (FATAL "| CORSIKA8 > The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif ()
# set CXX compile flags and options and warning settings
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_EXTENSIONS OFF)

#+++++++++++++++++++++++++++++
# Compiler and linker flags, settings
#
# enable warnings and disallow non-standard language
# configure the various build types here, too
# FYI: optimizer flags: -O2 would not trade speed for size, neither O2/3 use fast-math
# debug: O0, relwithdebinfo: 02, release: O3, minsizerel: Os (all defaults)
set (CMAKE_CXX_FLAGS "-Wall -pedantic -Wextra -Wno-ignored-qualifiers")
if (CORSIKA_SCL_CXX)
  add_definitions (-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

# clang produces a lot of unecessary warnings without this:
add_compile_options (
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-nonportable-include-path>
  )


#+++++++++++++++++++++++++++++
# Setup external dependencies.
#
###


#
# add cnpy temporarily. As long as SaveBoostHistogram does not save to parquet directly
#
add_subdirectory (externals/cnpy)

#+++++++++++++++++++++++++++++
# Coverage
#
# targets and settings needed to generate coverage reports
#if (CMAKE_BUILD_TYPE STREQUAL Coverage)

SET(COVERAGE_BUILD OFF CACHE BOOL "Activate coverage build")

if( COVERAGE_BUILD AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))


  message(INFO "\n|==========>  COVERAGE TARGET ACTIVATED.\n")

  set (c8_lcov_download_url "https://github.com/linux-test-project/lcov/releases/download")
  set (c8_lcov_version "1.16")
  set (c8_lcov_install_dir "${CMAKE_BINARY_DIR}/lcov")

  include(ExternalProject)
  ExternalProject_Add(lcov
    URL ${c8_lcov_download_url}/v${c8_lcov_version}/lcov-${c8_lcov_version}.tar.gz
    DOWNLOAD_NO_EXTRACT 0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND make -C <SOURCE_DIR> install PREFIX=${c8_lcov_install_dir}
  )

  find_package (Perl REQUIRED)

  # compile coverage under -O0 to avoid any optimization, function elimation etc.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 --coverage")

  # search for local lcov
  find_program (c8_lcov_bin lcov)
  if (NOT c8_lcov_bin)
    set (c8_lcov_bin "${c8_lcov_install_dir}/bin/lcov")
    message ("use C8 version of lcov ${c8_lcov_bin}")
  endif ()

  # search for local genhtml
  find_program (c8_genhtml_bin genhtml)
  if (NOT c8_genhtml_bin)
    set (c8_genhtml_bin "${c8_lcov_install_dir}/bin/genhtml")
    message ("use C8 version of genhtml ${c8_genhtml_bin}")
  endif ()

  set (GCOV gcov CACHE STRING "gcov executable" FORCE)


  # collect coverage data
  add_custom_command (
    OUTPUT raw-coverage.info
    COMMAND ${CMAKE_COMMAND} -E echo "Note: you need to run ctest at least once to generate the coverage data"
    COMMAND ${c8_lcov_bin} --gcov-tool=${GCOV} --rc lcov_branch_coverage=1
                           --directory . --capture --output-file raw-coverage.info
    )
  # remove uninteresting entries
  add_custom_command (
    OUTPUT coverage.info
    COMMAND ${c8_lcov_bin} -q --remove raw-coverage.info "*/usr/*" "/usr/*" --output-file coverage2.info
    COMMAND ${c8_lcov_bin} --remove coverage2.info
                        "*/externals/*" "*/tests/*" "*/sibyll2.3d.cpp" "*/.conan/*"
                        "*/include/Pythia8/*" "*/install/*"
                        "${CMAKE_SOURCE_DIR}/modules/*" "${CMAKE_BINARY_DIR}/modules/*"
                        "*/cxroot/*" "*/.conan2/*" "/boost/*"
                        --output-file coverage.info
    COMMAND ${CMAKE_COMMAND} -E remove coverage2.info
    DEPENDS raw-coverage.info
    )
  # generate html report
  add_custom_command (
    OUTPUT coverage-report
    COMMAND ${c8_genhtml_bin} --branch-coverage --show-details --title "CORSIKA 8 test coverage" --legend --demangle-cpp coverage.info -o coverage-report
    DEPENDS coverage.info
    )
  add_custom_target (coverage DEPENDS coverage-report)

endif ()

#+++++++++++++++++++++++++++++
# CTest config
#
enable_testing ()
if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
  list (APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
else (${CMAKE_VERSION} VERSION_LESS "3.12.0")
  set (CTEST_OUTPUT_ON_FAILURE 1) # this is for new versions of cmake
endif (${CMAKE_VERSION} VERSION_LESS "3.12.0")
set (CTEST_CUSTOM_COVERAGE_EXCLUDE "./tests/" "./examples/" "./modules/" "test*.(hpp/cpp)$")
# include this test only if NOT run on gitlab-ci; On CI this is a dedicated job:
if (NOT DEFINED ENV{CI})
  # add call to ./do-copyright.py to run as unit-test-case
  add_test (NAME copyright_notices COMMAND ./do-copyright.py WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif (NOT DEFINED ENV{CI})

#+++++++++++++++++++++++++++++
# Externals
#
set (Python_ADDITIONAL_VERSIONS 3)
find_package (PythonInterp 3 REQUIRED)

#+++++++++++++++++++++++++++++++
# exporting of CMake targets
#
include (CMakePackageConfigHelpers)

# list of targets that will be INSTALLed and EXPORTed
# Important: all those targets must be individually INSTALLed with "EXPORT CORSIKA8PublicTargets"
set (public_CORSIKA8_targets CORSIKA8)


#+++++++++++++++++++++++++++++
# CORSIKA8
#
add_library (CORSIKA8 INTERFACE)

set_target_properties (
  CORSIKA8
  PROPERTIES
  INTERFACE_CORSIKA8_MAJOR_VERSION 0
  COMPATIBLE_INTERFACE_STRING CORSIKA8_MAJOR_VERSION
  )

target_include_directories (
  CORSIKA8
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

# since CORSIKA8 is a header only library we must specify all link dependencies here:

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(CLI11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Arrow REQUIRED)
find_package(PROPOSAL REQUIRED)
find_package(Catch2 REQUIRED)

target_link_libraries (
  CORSIKA8
  INTERFACE
  BZip2::BZip2
  Boost::filesystem
  CLI11::CLI11
  Eigen3::Eigen
  spdlog::spdlog
  yaml-cpp::yaml-cpp
  Parquet::parquet_static
  PROPOSAL::PROPOSAL
  cnpy # for SaveBoostHistogram
  )

# "src" is needed, since some headers (namely GeneratedParticleProperties.inc ec.) are produced by python script from e.g. ParticleData.xml
add_subdirectory (src)
add_subdirectory (src/gpu/em/tables)
list (APPEND public_CORSIKA8_targets CORSIKA8GpuEmTables)
if (CORSIKA_ENABLE_CUDA)
  add_subdirectory (src/gpu/em)
  list (APPEND public_CORSIKA8_targets CORSIKA8GpuEm)
endif ()
add_subdirectory (documentation)

#+++++++++++++++++++++++++++++
#  = Add of subdirectories =
#

#+++++++++++++++++++++++++++++
# modules
#
set (CORSIKA_DATA_WITH_TEST ON) # we want to run the corsika-data unit test
add_subdirectory (modules/data) # this is corsika-data (submodule)
add_subdirectory (modules/common)
add_subdirectory (modules/pythia8)
add_subdirectory (modules/sibyll)
add_subdirectory (modules/sophia)
add_subdirectory (modules/qgsjetII)
add_subdirectory (modules/urqmd)
add_subdirectory (modules/conex)
add_subdirectory (modules/epos)
add_subdirectory (modules/tauola)
add_subdirectory (modules/fluka)

#
#+++++++++++++++++++++++++++++++
# standard applications
#

add_subdirectory(applications)

#+++++++++++++++++++++++++++++++
# unit testing
#
add_subdirectory (tests)

#+++++++++++++++++++++++++++++++
# installation
#
install (
  TARGETS CORSIKA8
  EXPORT CORSIKA8PublicTargets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
install (
  TARGETS CORSIKA8GpuEmTables
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
if (CORSIKA_ENABLE_CUDA)
  install (
    TARGETS CORSIKA8GpuEm
    EXPORT CORSIKA8PublicTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif ()
#
# header only part, just copy entire directory tree
#
install (DIRECTORY corsika DESTINATION include)
#
# generate cmake config package version (for find_package)
#
write_basic_package_version_file (
  ${PROJECT_BINARY_DIR}/corsikaConfigVersion.cmake
  VERSION ${c8_version}
  COMPATIBILITY SameMajorVersion
  )
#
# export targets in build tree (for find_package)
#
export (
  EXPORT CORSIKA8PublicTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/corsikaTargets.cmake"
  NAMESPACE CORSIKA8::
  )
#
# export targets in install tree (for find_package)
#
install (
  EXPORT CORSIKA8PublicTargets
  FILE corsikaTargets.cmake
  NAMESPACE CORSIKA8::
  DESTINATION lib/cmake/corsika
  )
 if(DEFINED CONAN_CMAKE_DIR)
 install (
  DIRECTORY conan_cmake/
  DESTINATION lib/cmake/dependencies
  )
 endif(DEFINED CONAN_CMAKE_DIR)
#
# config for build tree (for find_package)
#
configure_package_config_file (
  cmake/corsikaConfig.cmake.in
  ${PROJECT_BINARY_DIR}/corsikaConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
#
# corsikaDefines
#
configure_package_config_file (
  cmake/corsikaDefines.cmake
  ${PROJECT_BINARY_DIR}/corsikaDefines.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
#
# config for install tree  (for find_package)
# overwrite with install locations (if it was build as part of corsika)
#
set (Pythia8_INCDIR ${Pythia8_INCDIR_INSTALL})
set (Pythia8_LIBDIR ${Pythia8_LIBDIR_INSTALL})
configure_package_config_file (
  cmake/corsikaConfig.cmake.in
  ${PROJECT_BINARY_DIR}/corsikaConfig.cmake-install
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
#
# generate "corsika/corsika.hpp" file with cmake-level details and paths
# first for build-tree
#
# the location of the "corsika-data" is of fundamental importance
set (CORSIKA_CMAKE_DATA_DIR ${CMAKE_SOURCE_DIR}/modules/data)
configure_file (
  src/corsika.hpp.in
  ${PROJECT_BINARY_DIR}/corsika/corsika.hpp
)
#
# second also for install-tree
#
set (CORSIKA_CMAKE_DATA_DIR "${CMAKE_INSTALL_FULL_DATADIR}/corsika/data")
configure_file (
  src/corsika.hpp.in
  ${PROJECT_BINARY_DIR}/corsika.hpp-install
)
#
# installation of cmake and config files
#
install (
  FILES
  ${CMAKE_BINARY_DIR}/corsikaDefines.cmake
  ${CMAKE_BINARY_DIR}/corsikaConfigVersion.cmake
  DESTINATION lib/cmake/corsika
  )
#
# install and rename install-level corsikaConfig.cmake
#
install (
  FILES
  ${CMAKE_BINARY_DIR}/corsikaConfig.cmake-install
  RENAME corsikaConfig.cmake
  DESTINATION lib/cmake/corsika
  )
#
# also install-level config.hpp
#
install (FILES
  ${PROJECT_BINARY_DIR}/corsika.hpp-install
  RENAME corsika.hpp
  DESTINATION include/corsika
  )
#
# examples
#
install (DIRECTORY examples DESTINATION ${CMAKE_INSTALL_DATADIR}/corsika)
#
# data
#
install (DIRECTORY modules/data DESTINATION ${CMAKE_INSTALL_DATADIR}/corsika)
#
# tools
#
add_subdirectory (tools)


#+++++++++++++++++++++++++++++++
#
# final summary output
#
include (FeatureSummary)
feature_summary (WHAT ALL)
