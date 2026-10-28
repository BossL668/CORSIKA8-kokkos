SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable (c8_air_shower c8_air_shower.cpp)
target_link_libraries (c8_air_shower PRIVATE CORSIKA8::CascadeApplication)
add_executable (c8_ice_cascade c8_ice_cascade.cpp)
target_link_libraries (c8_ice_cascade PRIVATE CORSIKA8::CascadeApplication)
add_executable (c8_firn_cascade c8_firn_cascade.cpp)
target_link_libraries (c8_firn_cascade PRIVATE CORSIKA8::CascadeApplication)

set_target_properties(c8_air_shower c8_ice_cascade c8_firn_cascade PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/corsika"
)

if(WITH_FLUKA)
  message("compiling c8_air_shower.cpp with FLUKA")
  target_compile_definitions(c8_air_shower PRIVATE WITH_FLUKA)
else()
  message("compiling c8_air_shower.cpp with UrQMD")
endif()

install (
  TARGETS c8_air_shower c8_ice_cascade c8_firn_cascade DESTINATION bin
)
