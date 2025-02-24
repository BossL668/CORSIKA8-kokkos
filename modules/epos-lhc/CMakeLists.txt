set (
  MODEL_HEADERS
  epos.hpp
  )

enable_language (Fortran)

add_library (EPOS_LHC_static STATIC 
  epos-private.cpp
  epos-bas-lhc.f  epos-con-lhc.f eposm.f epos_j.f 
  epos-dky-lhc.f  epos-dro-lhc.f epos-ems-lhc.f  
  epos-fra-lhc.f  epos-ico-lhc.f epos-hnb-lhc.f 
  epos-ids-lhc.f  epos-int-lhc.f epos-jps-lhc.f  
  epos-omg-lhc.f  epos-par-lhc.f epos-sem-lhc.f 
  epos-rsh-lhc.f  epos-qsh-lhc.f epos-tim-lhc.f 
  epos-uti-lhc.f  epos-xan-lhc.f epos-xpr-lhc.f  
)
target_link_libraries(EPOS_LHC_static PUBLIC modules_common)

set_target_properties (
  EPOS_LHC_static
  PROPERTIES
  POSITION_INDEPENDENT_CODE 1
  CXX_VISIBILITY_PRESET hidden
  # unfortunately ignored by gfortran:
  Fortran_VISIBILITY_PRESET hidden
)

add_library(EPOS_LHC SHARED epos-public.cpp)
target_link_libraries(EPOS_LHC PRIVATE EPOS_LHC_static)
target_link_options(EPOS_LHC PRIVATE "LINKER:--exclude-libs,ALL") # hide symbols from static/private lib
target_include_directories (
  EPOS_LHC
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/corsika_modules/eposlhc>
)

install (
  FILES
  ${MODEL_HEADERS}
  DESTINATION include/corsika_modules/epos
  )

install (
  TARGETS EPOS_LHC
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION lib/corsika
  INCLUDES DESTINATION include/corsika_modules/epos
  )

# add EPOS to CORSIKA8 build
add_dependencies (CORSIKA8 EPOS_LHC)
target_link_libraries (CORSIKA8 INTERFACE EPOS_LHC)
