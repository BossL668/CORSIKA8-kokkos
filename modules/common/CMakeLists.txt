add_library (modules_common INTERFACE)


target_include_directories (
  modules_common
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/corsika_modules/common>
  )

install (
  FILES
  rng_decl.hpp rng_impl.hpp datadir.hpp
  DESTINATION include/corsika_modules/common
  )

install (
  TARGETS modules_common
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION lib/corsika
  LIBRARY DESTINATION lib/corsika # just for cmake 3.10.x (ubuntu 18)
  )

# add sibyll to corsika8 build
add_dependencies (CORSIKA8 modules_common)
target_link_libraries (CORSIKA8 INTERFACE modules_common)
