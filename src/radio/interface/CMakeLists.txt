add_library(CORSIKA8InterfaceRadio STATIC Output.cpp)
add_library(CORSIKA8::InterfaceRadio ALIAS CORSIKA8InterfaceRadio)
target_include_directories(CORSIKA8InterfaceRadio PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(CORSIKA8InterfaceRadio PUBLIC Kokkos::kokkos Eigen3::Eigen)
target_compile_features(CORSIKA8InterfaceRadio PUBLIC cxx_std_17)
set_target_properties(CORSIKA8InterfaceRadio PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME InterfaceRadio)
install(TARGETS CORSIKA8InterfaceRadio EXPORT CORSIKA8PublicTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
