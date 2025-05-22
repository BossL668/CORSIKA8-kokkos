SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable (c8_air_shower c8_air_shower.cpp)
target_link_libraries (c8_air_shower PRIVATE CORSIKA8)

if(WITH_FLUKA)
  message("compiling c8_air_shower.cpp with FLUKA")
  target_compile_definitions(c8_air_shower PRIVATE WITH_FLUKA)
else()
  message("compiling c8_air_shower.cpp with UrQMD")
endif()

install (
  TARGETS c8_air_shower DESTINATION bin
)

# for testing only!
add_executable(inttest inttest.cpp)
target_link_libraries(inttest PRIVATE QGSJetII fmt::fmt)
