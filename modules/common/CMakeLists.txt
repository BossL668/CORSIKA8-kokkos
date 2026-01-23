add_library (modules_common INTERFACE)


target_include_directories (
  modules_common
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/corsika_modules/common>
  )

set_target_properties (modules_common PROPERTIES
  PUBLIC_HEADER "rng_decl.hpp;rng_impl.hpp;datadir.hpp"
)

install (
  TARGETS modules_common
  EXPORT CORSIKA8PublicTargets
  PUBLIC_HEADER DESTINATION "include/corsika_modules/common"
)

# add sibyll to corsika8 build
add_dependencies (CORSIKA8 modules_common)
target_link_libraries (CORSIKA8 INTERFACE modules_common)
