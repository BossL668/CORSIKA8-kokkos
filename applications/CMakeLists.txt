# Keep installed applications relocatable, including when users exercise
# `cmake --install --prefix`.  All CORSIKA shared libraries installed by this
# directory tree live one level above bin/ in the configured GNU libdir.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

add_executable (c8_air_shower c8_air_shower.cpp)
target_link_libraries (c8_air_shower CORSIKA8)

# FLUKA discovery and application selection historically used two independent
# switches.  If a valid FLUKA library was discovered, make it the application
# default so a fresh build cannot silently select UrQMD instead.  An explicit
# -DWITH_FLUKA=OFF still remains available.
if (NOT DEFINED WITH_FLUKA)
  if (TARGET fluka)
    set (WITH_FLUKA ON CACHE BOOL
         "Use FLUKA for low-energy hadronic interactions")
  else ()
    set (WITH_FLUKA OFF CACHE BOOL
         "Use FLUKA for low-energy hadronic interactions")
  endif ()
endif ()

if (WITH_FLUKA AND NOT TARGET fluka)
  message (FATAL_ERROR
           "WITH_FLUKA=ON, but no valid FLUKA library was configured. "
           "Set C8_FLUKALIB or FLUPRO to a directory containing libflukahp.a.")
endif ()

# A Conda Fortran compiler can contribute an old-sysroot libm linker script
# even when the final C++ driver and all C++ objects use the host glibc. Put
# the matching host multiarch directory first instead of applying the Conda
# sysroot to host objects.
if (CMAKE_Fortran_COMPILER_SYSROOT)
  set (_c8_host_multiarch_libdir
       "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
  if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options (
      c8_air_shower PRIVATE
      "-L${_c8_host_multiarch_libdir}"
      )
  endif ()
endif ()

if (CORSIKA_ENABLE_CUDA)
  target_link_libraries (c8_air_shower CORSIKA8GpuEm)
  target_compile_definitions (c8_air_shower PRIVATE CORSIKA8_WITH_CUDA_EM=1)

  add_executable (cuda_decision_replay cuda_decision_replay.cpp)
  target_link_libraries (
    cuda_decision_replay
    PRIVATE
    CORSIKA8
    CORSIKA8GpuEm
    )
endif ()

add_executable (gpu_em_tablegen gpu_em_tablegen.cpp)
target_link_libraries (
  gpu_em_tablegen
  PRIVATE
  CORSIKA8GpuEmTables
  CLI11::CLI11
  PROPOSAL::PROPOSAL
  spdlog::spdlog
  )
add_dependencies (gpu_em_tablegen GenParticlesHeaders GenMediaProperties)

if(WITH_FLUKA)
  message("compiling c8_air_shower.cpp with FLUKA")
  target_compile_definitions(c8_air_shower PRIVATE WITH_FLUKA)
  add_executable (fluka_batch_worker fluka_batch_worker.cpp)
  target_link_libraries (fluka_batch_worker PRIVATE CORSIKA8 fluka)
else()
  message("compiling c8_air_shower.cpp with UrQMD")
endif()

install (
  TARGETS c8_air_shower gpu_em_tablegen DESTINATION bin
)

if (TARGET cuda_decision_replay)
  install (TARGETS cuda_decision_replay DESTINATION bin)
endif ()

if (TARGET fluka_batch_worker)
  install (TARGETS fluka_batch_worker DESTINATION bin)
endif ()
